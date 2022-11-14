
import os
from tkinter import S
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model import RELATION
from functions import LatentLoss, DiffLoss, SimLoss
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import torch.nn.functional as F

import argparse


parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('-e', '--epoches', type=int, default=150)
parser.add_argument('-s', '--steps', type=int, default=5000)
parser.add_argument('-t', '--target', type=str, default='cdk2')
parser.add_argument('-b', '--batchsize', type=int, default=256)
parser.add_argument('-d', '--device', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.device

model_root = 'relation'
cuda = True
cudnn.benchmark = True

#######################################
# params of relation model            #
#######################################

lr = 1e-4
step_decay_weight = 0.95
lr_decay_step = 20000
savedir='./output/'

###########################################################
# the α，β，γ weight of the dofferent relation loss parts，#
###########################################################

alpha_weight = 1
beta_weight = 0.075
gamma_weight = 0.05



###################################################
# load the 3D-grid format of                      #
# source (ligand-only) and                        #
# target data(ligand-protein complexes)           #
###################################################

class relation_dataset(data.Dataset):
    def __init__(self,path):
        self.source=np.load(path)
        self.representation =self.source['dataset']
        self.captions= self.source['smi']
        self.smi_length = []
        for i in self.source['smi']:
            i=i.tolist()
            x = i.index(2)
            self.smi_length.append(x)



    def __getitem__(self,index):
        grid = self.representation[index]
        caption = self.captions[index]
        length = self.smi_length[index]

        return grid, caption, length

    
    def __len__(self):
        return len(self.captions)

target_dataset=relation_dataset('/home/mingyang/debug/RELATION/grid/%s_pkis.npz' %(args.target))
#source_dataset=relation_dataset('/home/mingyang/debug/RELATION/grid/%s_pkis.npz' %(args.target))
source_dataset=relation_dataset('/home/mingyang/debug/RELATION/grid/zinc/zinc_900000-1000000.npz')




dataloader_source = torch.utils.data.DataLoader(dataset=source_dataset,batch_size=args.batchsize,shuffle=True)
dataloader_target = torch.utils.data.DataLoader(dataset=target_dataset,batch_size=args.batchsize,shuffle=True)





relation = RELATION()


##########################################
# Decay learning rate by a factor of     #
# step_decay_weight every lr_decay_step  #
##########################################
def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer

#############################
# setup the adamoptimizer   #
#############################

optimizer = optim.Adam(relation.parameters(), lr=lr)


####################################
# setup the relation loss formula  #
####################################
loss_diff = DiffLoss()
loss_similarity = SimLoss()
loss_latent = LatentLoss()
loss_caption = nn.CrossEntropyLoss()



if cuda:
    relation = relation.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()
    loss_latent = loss_latent.cuda()
    loss_caption = loss_caption.cuda()
for p in relation.parameters():
    p.requires_grad = True

current_step = 0
log_file = open(os.path.join(savedir, "log.txt"), "w")


##########################
#RELATION network tarinng#
##########################

for epoch in tqdm(range(args.epoches)):
    

    s = 0



    while s < args.steps:

        ###################################
        # target data training            #
            ###################################
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        data_target = data_target_iter.__next__()

        relation.zero_grad()
        
        loss = 0
        


        data_target_grid, data_target_caption, data_target_length = data_target
        data_target_caption=data_target_caption.long()
        data_target_caption_input=[]
        for i in data_target_caption:
            i = i.numpy()
            index = np.argwhere(i==2)
            i=i[1:int(index[0])]
            i=torch.from_numpy(i)
            data_target_caption_input.append(i)




        if cuda:
            data_target_grid = data_target_grid.cuda()
            data_target_caption = data_target_caption.cuda()
        

        data_target_grid, data_target_caption= Variable(data_target_grid), Variable(data_target_caption)




        result = relation(input_data=data_target_grid, input_caption=data_target_caption_input, mode='target', rec_scheme='all', caption=data_target_caption, lengths=data_target_length)
        
        _, _, _, target_mu, target_logvar, target_privte_code,_,target_share_code,_ =result
        


        tar_latloss = gamma_weight * loss_latent(target_mu, target_logvar)
        loss += tar_latloss
        tar_diffloss = beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += tar_diffloss


        ###################################
        # source data training            #
        ###################################

        data_source = data_source_iter.__next__()

        relation.zero_grad()


        data_source_grid, data_source_caption, data_source_length = data_source
        data_source_caption=data_source_caption.long()
        data_source_caption_input=[]
        for i in data_source_caption:
            i = i.numpy()
            index = np.argwhere(i==2)
            i=i[1:int(index[0])]
            i=torch.from_numpy(i)
            data_source_caption_input.append(i)

        if cuda:
            data_source_grid = data_source_grid.cuda()
            data_source_caption = data_source_caption.cuda()

        data_source_grid, data_source_caption= Variable(data_source_grid), Variable(data_source_caption)

        result = relation(input_data=data_source_grid, input_caption=data_source_caption_input, mode='source', rec_scheme='all', caption=data_source_caption, lengths=data_source_length)
        _, _, _, source_mu, source_logvar, source_privte_code,_,source_share_code, source_recon_code =result

        bi_sim = gamma_weight * loss_similarity(source_share_code, target_share_code)
        loss += bi_sim

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff
        source_latloss = gamma_weight *  loss_latent(source_mu, source_logvar)
        loss = source_latloss



        x = nn.utils.rnn.pad_sequence(data_source_caption_input, batch_first=True,padding_value=0)
        x=x[:, 1:].contiguous().view(-1)
        x=x.cuda()
        y=source_recon_code[:, :-1].contiguous().view(-1, source_recon_code.size(-1))

        source_simse = alpha_weight * F.cross_entropy(
            y,
            x,
            ignore_index=0
        )
        
        loss += source_simse
        print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()
        s += 1
        current_step += 1

    run_log = "epoch %s, loss: %.5f" %(
        str(epoch+1),
        float(loss.data.cpu().numpy())       
        )

    
    log_file.write(run_log + "\n")
    log_file.flush()

    print(run_log)
    if (epoch+1) % 10 == 0:
        torch.save(relation.state_dict(),os.path.join(savedir,'rel-%s-%d.pth' % (args.target,epoch+1)))

print ('The job is done ')





