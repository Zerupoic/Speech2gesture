import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch.autograd import Variable

from speech2gesture import *
from pats_master.data import Data
from standardizational_index import get_mean_std, get_mean_std_necksub
from funcs import pos_to_motion
from evaluation import compute_pck, compute_pck_radius
# from randpos_generator import randpos_gen


common_kwargs = dict(path2data = '../pats/data',
                     speaker = ['almaram'],
                     modalities = ['pose/data', 'audio/log_mel_512'],
                     fs_new = [15, 15],
                     batch_size = 4,
                     window_hop = 5)

dataloader = Data(**common_kwargs)



### testing
pck_sum = 0.
L1_loss_sum = 0.
criterion = nn.L1Loss()
pose_mean, pose_std = get_mean_std_necksub(dataloader)
mean_pose = pose_mean.reshape(1, 64, 104)
with torch.no_grad():
    for ibatch, batch in enumerate(dataloader.test, 1):
        real_pose = batch['pose/data']
        real_pose_norm = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)
        real_neck = real_pose_norm[:, :, :, 0].reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], 2, 1)
        real_pose_norm = torch.sub(real_pose_norm, real_neck)
        real_pose_norm = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], -1)
        real_pose_norm = torch.sub(real_pose_norm, pose_mean)
        real_pose_norm = torch.div(real_pose_norm, pose_std)

        # rand_pose = randpos_gen(pose_mean, pose_std, batch=real_pose.shape[0], frames=real_pose.shape[1])
        
        L1_loss_sum += criterion(real_pose_norm, mean_pose)

        mean_pose_xy = mean_pose.reshape(mean_pose.shape[0], mean_pose.shape[1], 2, -1)
        gp_torch = torch.flatten(mean_pose_xy, start_dim=0, end_dim=1)
        gp = gp_torch.numpy()
        real_pose_xy = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose.shape[1], 2, -1)
        rp_torch = torch.flatten(real_pose_xy, start_dim=0, end_dim=1)
        rp = rp_torch.numpy()

        pck = compute_pck(gp, rp)
        pck_sum += np.mean(pck, axis=0)

        
print('average pck: ', pck_sum / ibatch)
print('average L1 loss: ', L1_loss_sum / ibatch)
