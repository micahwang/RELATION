import torch
import numpy as np
from model import RELATION
import pandas as pd
import rdkit
from rdkit import Chem
from utils import autodock
import tqdm
import sparse_gp as SparseGP
from utils import save_object
import pyscreener as ps

voc_set=['pad', 'bos', 'eos', '5', 'Y', ')', 'Z', '[', ']', '-', 
    'S', '1', 'O', 'N', "'", ' ', 'C', '(', 'n', 'c', '#', 's', '6', 
    'X', '4', ',', '2', 'o', 'F', '=', '3', '.', 'I', '/', '+', '\\', '@', 'H', 'P']

vocab_i2c_v1 = {i: x for i, x in enumerate(voc_set)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

def decode_smiles(tok):
    gen_smi = []
    for i in tok:
        csmile = ""
        for xchar in i[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smi.append(csmile)
    return gen_smi


def docking(smi):
    metadata = ps.build_metadata("vina")
    docking = ps.virtual_screen("vina", [".data/docking/4gv1_pre.pdb"], (-18.2, 14.4, -16.1), (15.4, 13.9, 14.5), metadata, ncpu=24) {...}
    scores = docking(smi)
    return scores



def greed_search(n_sampled,path):
    relation = RELATION()
    relation.load_state_dict(torch.load('./model.pth'))
    relation.to('cuda')
    relation.eval()


    g_smi=[]
    for i in range(n_sampled):
        z_prior = torch.randn(1,512).to('cuda')
        gen_tok = relation.sample(z_prior)
        gen_smi = decode_smiles(gen_tok)
        g_smi += gen_smi

    pd.DataFrame(g_smi).to_csv(path,index=0)    



def main_bo(n_sampled=500,iterations=5,random_seed=1):
    

    relation = RELATION()
    relation.load_state_dict(torch.load('./model.pth'))
    relation.to('cuda')
    relation.eval()

    np.random.seed(random_seed)
    
    X = np.loadtxt('./data/bo/latent_points.npz')['arr_0']
    y = -np.loadtxt('./data/bo/docking_scores.npz')['akt1']
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
        
        save_object(gen_smi, './output' + "/scores{}.dat".format(iteration))
        save_object(scores, './output' + "/next_inputs{}.dat".format(iteration))       
      
        X_train = np.concatenate([X_train, new_latentpoint], 0)
        y_train = np.concatenate([y_train, np.array(scores)[:, None]], 0)




        iteration += 1


if __name__ == '__main__':

    main_bo(n_sampled=500,iterations=5)




