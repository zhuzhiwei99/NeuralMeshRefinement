

import torch
import sys
import os



def nndistance_simple(rec, data):
    """
    A simple nearest neighbor search, not very efficient, just for reference
    rec: B x N x 3
    data: B x M x 3
    data_dist: B x N
    rec_dist: B x M
    data_idx: B x N
    rec_idx: B x M
    """
    rec_sq = torch.sum(rec * rec, dim=2, keepdim=True)
    data_sq = torch.sum(data * data, dim=2, keepdim=True)
    cross = torch.matmul(data, rec.permute(0, 2, 1))
    dist = data_sq - 2 * cross + rec_sq.permute(0, 2, 1)
    data_dist, data_idx = torch.min(dist, dim=2)
    rec_dist, rec_idx = torch.min(dist, dim=1)
    return data_dist, rec_dist, data_idx, rec_idx


try:
    # If you want to use the efficient NN search for computing CD loss, compiled the nndistance()
    # function under the third_party folder according to instructions in Readme.md
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../third_party/nndistance'))
    from modules.nnd import NNDModule
    nndistance = NNDModule()
except ModuleNotFoundError:
    # Without the compiled nndistance(), by default the nearest neighbor will be done using pytorch-geometric
    nndistance = nndistance_simple

class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, rec, data):
        '''
        x: (B, N, 3)
        y: (B, M, 3)
        '''
        data_dist, rec_dist, data_idx, rec_idx = nndistance(rec.contiguous(), data.contiguous())
        dist = torch.max(torch.mean(data_dist, dim=1), torch.mean(rec_dist, dim=1), dim=1)  # (B,)
        dist = torch.mean(dist)
        return dist


class NormalsLoss(torch.nn.Module):
    def __init__(self):
        super(NormalsLoss, self).__init__()

    def forward(self, rec, data, rec_normals, data_normals):
        '''
        x: (B, N, 3)
        y: (B, M, 3)
        x_normals: (B, N, 3)
        y_normals: (B, M, 3)
        '''

        data_dist, rec_dist, data_idx, rec_idx = nndistance(rec.contiguous(), data.contiguous())
        data_rec_normals = data_normals[:,data_idx].squeeze(0)
        normals_loss = torch.norm(data_rec_normals - rec_normals, dim=2) # (B, N)
        normals_loss = torch.mean(torch.mean(normals_loss, dim=1))
        return normals_loss