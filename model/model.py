import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


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
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(1024, 39)
        self.embedding = nn.Embedding(39, 512)
        self.lstm = nn.LSTM(512, 1024, 3, batch_first=True)
        
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

    def decode(self, z, captions, lengths):
        embedding = self.embedding(captions)
        embedding = torch.cat((z.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embedding, lengths, batch_first=True,enforce_sorted=False)
        output, hn = self.lstm(packed)
        y = self.fc4(output[0])
        return y        




    def dis_aae(self, z):
        discri=[]
        discri.append(nn.Linear(512,256))
        discri.append(nn.Linear(256,128))
        discri.append(nn.Linear(128,64))
        discri.append(nn.Linear(64,2))
        discriminator =  nn.Sequential(*discri)    
        return discriminator(z)



    def forward(self, input_data, mode, rec_scheme, caption, lengths):
        result=[]

        if mode == 'source':

            # source private encoder
            x = self.pri_source_encoder(input_data)
            x = x.mean(2).mean(2).mean(2)       
            pri_feat_mu, pri_feat_logvar = self.fc11(x), self.fc12(x)
            pri_latent = self.reparametrize(pri_feat_mu, pri_feat_logvar)


        elif mode == 'target':

            # target private encoder
            x = self.pri_target_encoder(input_data)
            x = x.mean(2).mean(2).mean(2)       
            pri_feat_mu, pri_feat_logvar = self.fc11(x), self.fc12(x)
            pri_latent = self.reparametrize(pri_feat_mu, pri_feat_logvar)

        result.extend([pri_feat_mu, pri_feat_logvar, pri_latent])

        # shared encoder
        x = self.shared_encoder(input_data)
        x = x.mean(2).mean(2).mean(2) 
        shr_feat_mu, shr_feat_logvar = self.fc11(x), self.fc12(x)
        shr_latent = self.reparametrize(shr_feat_mu, shr_feat_logvar)
        result.append(shr_latent)

        # shared decoder

        if rec_scheme == 'share':
            union_latent = shr_latent
        elif rec_scheme == 'all':
            union_latent = pri_latent + shr_latent
        elif rec_scheme == 'private':
            union_latent = pri_latent
        
        
        re_smi = self.decode(union_latent,caption,lengths)
        result.append(re_smi)
        

        return result
        
    def sample(self,z):
        sampled_ids = []
        inputs=z.unsqueeze(1)
        inputs=inputs[0,:,:]  
        for i in range(120):
            latt, _ = self.lstm(inputs)
            outputs = self.fc3(latt.squeeze(1))
            toks = outputs.max(1)[1]
            sampled_ids.append(toks)
            inputs = self.embedding(toks)
            inputs = inputs.unsqueeze(1)
        return sampled_ids







