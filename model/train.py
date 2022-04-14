
import os
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
import tqdm




model_root = 'relation'
cuda = True
cudnn.benchmark = True

#######################################
# params of relation model            
#######################################

lr = 1e-2
n_epoch = 150
step_decay_weight = 0.95
lr_decay_step = 20000
active_domain_loss_step = 10000
savedir='./output/'

#####################################################################################################
# the α，β，γ weight of the different relation loss parts，
###########################################################
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
i


log_file = open(os.path.join(savedir, "log.txt"), "w")

###################################################################################################
#    Implement the dataloader for source and target dataset.                                              
#      This is used to train the relation captioning model.    
#                                            
###################################################################################################

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

#####################################################################
#The 3 special tokens: grid, caption and length 
####################################################################

    def __getitem__(self,index):
        grid = self.representation[index]
        caption = self.captions[index]
        length = self.smi_length[index]

        return grid, caption, length

    
    def __len__(self):
        return len(self.captions)


source_dataset=relation_dataset('data/zinc/zinc.npz')
target_dataset=relation_dataset('data/akt1/akt_pkis.npz')



dataloader_source = torch.utils.data.DataLoader(dataset=source_dataset,batch_size=32,shuffle=True)
dataloader_target = torch.utils.data.DataLoader(dataset=target_dataset,batch_size=32,shuffle=True)





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



##########################
#RELATION network tarinng#
##########################

for epoch in range(n_epoch):
    

    i = 0

    while i < 30000:

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

        if cuda:
            data_target_grid = data_target_grid.cuda()
            data_target_caption = data_target_caption.cuda()
        

        data_target_grid, data_target_caption  = Variable(data_target_grid), Variable(data_target_caption)




        result = relation(input_data=data_target_grid, mode='target', rec_scheme='all', caption=data_target_caption, lengths=data_target_length)
        target_mu, target_logvar, target_privte_code, target_share_code, _ = result

        tar_latloss = loss_latent(target_mu, target_logvar)
        loss += tar_latloss
        tar_diffloss = beta_weight * loss_diff(target_privte_code, target_share_code)
    

        optimizer.step()

        ###################################
        # source data training            #
        ###################################

        data_source = data_source_iter.__next__()

        relation.zero_grad()


        data_source_grid, data_source_caption, data_source_length = data_source
        data_source_caption=data_source_caption.long()

        if cuda:
            data_source_grid = data_source_grid.cuda()
            data_source_caption = data_source_caption.cuda()

        data_source_grid, data_source_caption  = Variable(data_source_grid), Variable(data_source_caption)


        result = relation(input_data=data_source_grid, mode='source', rec_scheme='all', caption=data_source_caption, lengths=data_source_length)
        source_mu, source_logvar, source_privte_code, source_share_code, source_recon_code = result

        bi_sim = gamma_weight * loss_similarity(source_share_code, target_share_code)
        loss += bi_sim

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff
        targets = pack_padded_sequence(data_source_caption, data_source_length, batch_first=True,enforce_sorted=False)[0]

        source_simse = alpha_weight * loss_caption(source_recon_code, targets)
        loss += source_simse

        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        i += 1
        current_step += 1

    result = "Step: {}, relation_loss: {:.5f}, ".format(i + 1,float(loss.data.cpu().numpy()) if type(loss) != float else 0.)
    log_file.write(result + "\n")
    log_file.flush()


    if epoch % 10 == 0:
        torch.save(relation.state_dict(),os.path.join(savedir,'relation-akt1-%d.pth' % (i + 1)))



    print ('step: %d, loss: %f' % (current_step, loss.cpu().data.numpy()))
    torch.save(relation.state_dict(), os.path.join(savedir,'relation-akt1-%d.pth' % (epoch + 1)))

print ('The job is done ')





