from torch.autograd import Function
import torch.nn as nn
import torch



class LatentLoss(nn.Module):    
    def __init__(self):
        super(LatentLoss, self).__init__()

    def forward(self, mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        latentLoss = torch.sum(KLD_element).mul_(-0.5)
        return latentLoss





class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, re_x, x):
        diffs = torch.add(x, -re_x)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, re_x, x):
        diffs = torch.add(x, - re_x)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, x, y):

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)

        x_l2_norm = torch.norm(x, p=2, dim=1, keepdim=True).detach()
        x_l2 = x.div(x_l2_norm.expand_as(x) + 1e-6)

        y_l2_norm = torch.norm(y, p=2, dim=1, keepdim=True).detach()
        y_l2 = y.div(y_l2_norm.expand_as(y) + 1e-6)

        diff_loss = torch.mean((x_l2.t().mm(y_l2)).pow(2))

        return diff_loss

class SimLoss(nn.Module):    
    def __init__(self):
        super(SimLoss, self).__init__()


    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def forward(self,x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd




