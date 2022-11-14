import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RELATION(nn.Module):
    def __init__(self):
        super(RELATION, self).__init__()

        # private source encoder(Ligand in ZINC dataset)
        pri_sou_enc = []
        pri_tar_enc = []
        shr_enc = []
        self.fc11 = nn.Linear(256,512)
        self.fc12 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,1024)
        self.fc4 = nn.Linear(1024, 39)
        self.gru1=nn.GRU(39, 256, 3, batch_first=True)
        self.embedding = nn.Embedding(39, 39, 0)
        self.gru=nn.GRU(551, 1024, 3, batch_first=True)
        
        in_channels=19
        out_channels = 32
        for i in range(8):
            pri_sou_enc.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            pri_sou_enc.append(nn.BatchNorm3d(out_channels))
            pri_sou_enc.append(nn.ReLU())
            in_channels=out_channels

            if (i+1) % 2 ==0: 
                out_channels *= 2
                pri_sou_enc.append(nn.MaxPool3d((2, 2, 2)))
        pri_sou_enc.pop()
        self.pri_source_encoder=nn.Sequential(*pri_sou_enc)
        

        in_channels=19
        out_channels = 32
        for i in range(8):
            pri_tar_enc.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            pri_tar_enc.append(nn.BatchNorm3d(out_channels))
            pri_tar_enc.append(nn.ReLU())
            in_channels=out_channels
            if (i+1) % 2 ==0:
                out_channels *= 2
                pri_tar_enc.append(nn.MaxPool3d((2, 2, 2)))
        pri_tar_enc.pop()
        self.pri_target_encoder=nn.Sequential(*pri_tar_enc)

        in_channels=19
        out_channels = 32
        for i in range(8):
            shr_enc.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            shr_enc.append(nn.BatchNorm3d(out_channels))
            shr_enc.append(nn.ReLU())
            in_channels=out_channels
            if (i+1) % 2 ==0:
                out_channels *= 2
                shr_enc.append(nn.MaxPool3d((2, 2, 2)))
        shr_enc.pop()
        self.shared_encoder=nn.Sequential(*shr_enc)




    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        #eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = torch.FloatTensor(std.size()).normal_()
        #eps = Variable(eps)
        eps = Variable(eps).cuda()
        return (eps.mul(std)).add_(mu)


    def decode(self, z, captions):
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True,padding_value=0)
        lengths=[len(i_x) for i_x in captions]
        x_emb = self.embedding(captions)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = pack_padded_sequence(x_input, lengths,batch_first=True)
        h_0 = self.fc3(z)
        h_0 = h_0.unsqueeze(0).repeat(3, 1, 1)
        output, _ = self.gru(x_input,h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.fc4(output)
        return y



    def forward(self, input_data, input_caption, mode, rec_scheme, caption, lengths):
        result=[]

        if mode == 'source':

            # source private encoder
            x = self.pri_source_encoder(input_data)
            x = x.mean(2).mean(2).mean(2)       
            pri_feat_mu, pri_feat_logvar = self.fc11(x), self.fc12(x)
            pri_latent = self.reparametrize(pri_feat_mu, pri_feat_logvar)
            
            x_cap= [self.embedding(i_c.cuda()) for i_c in input_caption]
            x_cap = nn.utils.rnn.pack_sequence(x_cap,enforce_sorted=False)
            _,h =self.gru1(x_cap)
            h = h[-(1 + int(self.gru1.bidirectional)):]
            h = torch.cat(h.split(1), dim=-1).squeeze(0)
            cap_mu, cap_logvar = self.fc11(h), self.fc12(h)           
            cap_pri_latent = self.reparametrize(cap_mu, cap_logvar)



        elif mode == 'target':

            # target private encoder
            x = self.pri_target_encoder(input_data)
            x = x.mean(2).mean(2).mean(2)       
            pri_feat_mu, pri_feat_logvar = self.fc11(x), self.fc12(x)
            pri_latent = self.reparametrize(pri_feat_mu, pri_feat_logvar)

            x_cap= [self.embedding(i_c.cuda()) for i_c in input_caption]
            x_cap = nn.utils.rnn.pack_sequence(x_cap,enforce_sorted=False)
            _,h =self.gru1(x_cap)
            h = h[-(1 + int(self.gru1.bidirectional)):]
            h = torch.cat(h.split(1), dim=-1).squeeze(0)
            cap_mu, cap_logvar = self.fc11(h), self.fc12(h)           
            cap_pri_latent = self.reparametrize(cap_mu, cap_logvar)

        result.extend([pri_feat_mu, pri_feat_logvar, pri_latent, cap_mu, cap_logvar,cap_pri_latent])

        # shared encoder
        x = self.shared_encoder(input_data)
        x = x.mean(2).mean(2).mean(2) 
        shr_feat_mu, shr_feat_logvar = self.fc11(x), self.fc12(x)
        shr_latent = self.reparametrize(shr_feat_mu, shr_feat_logvar)
        x_cap= [self.embedding(i_c.cuda()) for i_c in input_caption]
        x_cap = nn.utils.rnn.pack_sequence(x_cap,enforce_sorted=False)
        _,h =self.gru1(x_cap)
        h = h[-(1 + int(self.gru1.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        cap_mu, cap_logvar = self.fc11(h), self.fc12(h)           
        cap_shr_latent = self.reparametrize(cap_mu, cap_logvar)




        result.extend([shr_latent,cap_shr_latent])

        # shared decoder

        if rec_scheme == 'share':
            union_latent = cap_shr_latent
        elif rec_scheme == 'all':
            union_latent = cap_shr_latent + cap_pri_latent
        elif rec_scheme == 'private':
            union_latent = cap_pri_latent
        
        #re_smi = self.decode(cap_pri_latent,caption,lengths)
        cap= [i_c.cuda() for i_c in input_caption]
        re_smi = self.decode(union_latent,cap)
        result.append(re_smi)
        

        return result
        
    def sample(self,z,n_sample,max_len,dev):
        z_0 = z.unsqueeze(1)
        h=self.fc3(z)
        h = h.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        w = torch.tensor(1, device=dev).repeat(n_sample)#1:bos sn
        x = torch.tensor([0], device=dev).repeat(n_sample,max_len)#0:pad sn
        x[:,0]=1#1:bos sn
        end_pads = torch.tensor([max_len], device=dev).repeat(n_sample)
        eos_mask = torch.zeros(n_sample, dtype=torch.uint8,device=dev)
        for i in range(1, max_len):
            x_emb = self.embedding(w).unsqueeze(1)
            x_input = torch.cat([x_emb, z_0], dim=-1)

            o, h = self.gru(x_input, h)
            y = self.fc4(o.squeeze(1))
            y = torch.nn.functional.softmax(y, dim=-1)
            w = torch.multinomial(y, 1)[:, 0]
            x[~eos_mask, i] = w[~eos_mask]
            i_eos_mask = ~eos_mask & (w == 2)
            end_pads[i_eos_mask] = i + 1
            eos_mask = eos_mask | i_eos_mask
        gen_id = []
        for i in range(x.size(0)):
            gen_id.append(x[i, :end_pads[i]])

        return gen_id


