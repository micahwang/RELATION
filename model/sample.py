import torch
import numpy as np
from model import RELATION
import pandas as pd
import rdkit
from rdkit import Chem
import tqdm
import sparse_gp as SparseGP
from utils import save_object
import pyscreener as ps
import argparse



parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('-m', '--method', type=int, default=0)
parser.add_argument('-n', '--numbers', type=int, default=500)
parser.add_argument('-p', '--path', type=str, default='./output/')
parser.add_argument('-l', '--max_length', type=int, default=80)
parser.add_argument('-t', '--target', type=str, default='cdk2')
parser.add_argument('-d', '--device', type=str, default='0')
parser.add_argument('-i', '--iter', type=int, default=5)


args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES']=args.device

voc_set=['pad', 'bos', 'eos', '5', 'Y', ')', 'Z', '[', ']', '-', 
    'S', '1', 'O', 'N', "'", ' ', 'C', '(', 'n', 'c', '#', 's', '6', 
    'X', '4', ',', '2', 'o', 'F', '=', '3', '.', 'I', '/', '+', '\\', '@', 'H', 'P']

vocab_i2c_v1 = {i: x for i, x in enumerate(voc_set)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

def decode_smiles(tok):
    gen_smi = []
    for i in tok:
        i=i.tolist()
        print(i)
        csmile = ''
        for xchar in i[1:]:
            if xchar == 2 or xchar == 0:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smi.append(csmile)
    return gen_smi



def gen_smi(n_sample,path,max_len,target):
    relation = RELATION()
    #relation.load_state_dict(torch.load(os.path.join(path, target+".pth"),map_location=dev))
    relation.load_state_dict(torch.load('/home/mingyang/debug/RELATION_1/output/rel-'+target+'.pth'))
    relation.to('cuda')
    relation.eval()
    z_prior = torch.randn(n_sample,512).to('cuda')
    gen_tok = relation.sample(z_prior,n_sample,max_len,dev='cuda')
    gen_list=decode_smiles(gen_tok)
        

    pd.DataFrame({'SMILES':gen_list}).to_csv(os.path.join(path, "gen_smi.csv"),index=0)    


def main_bo(n_sampled=500,iterations=5,random_seed=1,path='./',target='akt1'):
    

    relation = RELATION()
    relation.load_state_dict(torch.load('./model.pth'))
    relation.to('cuda')
    relation.eval()

    np.random.seed(random_seed)
    
    X = np.loadtxt('./data/bo/latent_points.npz')['arr_0']
    y = -np.loadtxt('./data/bo/docking_scores.npz')[target]
    y = y.reshape((-1, 1))

    n = X.shape[0]
    permutation = np.random.choice(n, n, replace=False)

    X_train = X[permutation, :][0:np.int(np.round(0.9 * n)), :]
    X_test = X[permutation, :][np.int(np.round(0.9 * n)):, :]

    y_train = y[permutation][0: np.int(np.round(0.9 * n))]
    y_test = y[permutation][np.int(np.round(0.9 * n)):]

    np.random.seed(random_seed)



    iteration = 0
    while iteration < iterations:
        # We fit the GP
        np.random.seed(iteration * random_seed)
        M = 500
        sgp = SparseGP(X_train, 0 * X_train, y_train, M)
        sgp.train_via_ADAM(X_train,
                           0 * X_train,
                           y_train,
                           X_test,
                           X_test * 0,
                           y_test,
                           minibatch_size=10 * M,
                           max_iterations=5,
                           learning_rate=0.001)



        next_inputs = sgp.batched_greedy_ei(n_sampled,
                                            np.min(X_train, 0),
                                            np.max(X_train, 0))
        gen_smi = []
        new_latentpoint = []
        scores=[]

        for i in tqdm(range(n_sampled)):
            grid = next_inputs[i]
            s = relation.sample(grid)
            smi = decode_smiles(s)
            gen_smi.append(smi)
            docking_score=0 if Chem.MolFromSmiles(smi) is None else docking(smi)
            if docking_score <= -8:
                new_latentpoint.append(grid)
            scores.append(docking_score)    
        new_latentpoint = next_inputs
        new_latentpoint = np.vstack(new_latentpoint) 
        
        save_object(gen_smi, path + "/gen{}.dat".format(iteration))
  
      
        X_train = np.concatenate([X_train, new_latentpoint], 0)
        y_train = np.concatenate([y_train, np.array(scores)[:, None]], 0)




        iteration += 1







if __name__ == '__main__':

    

    if args.method == 1:
        main_bo(n_sampled=args.numbers,iterations=args.iter,random_seed=1, path=args.path,target=args.target)
    else:
        gen_smi(n_sample=args.numbers,path=args.path,max_len=args.max_length,target=args.target)


