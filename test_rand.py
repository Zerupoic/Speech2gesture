import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from random import choice

from torch.autograd import Variable

from speech2gesture import *
from pats_master.data import Data
from standardizational_index import get_mean_std, get_mean_std_necksub
from funcs import pos_to_motion
from evaluation import compute_pck, compute_pck_radius
from randpos_generator import randpos_gen

# MODEL_PATH_G = './save/gen_almaram_norm'


common_kwargs = dict(path2data = '../pats/data',
                     speaker = ['lec_cosmic'],
                     modalities = ['pose/data'],
                     fs_new = [15, 15],
                     batch_size = 1,
                     window_hop = 5)

dataloader = Data(**common_kwargs)



# ### testing
# pck_sum_rand = 0.
# pck_sum_mean = 0.
# L1_loss_sum_rand = 0.
# L1_loss_sum_mean = 0.
# criterion = nn.L1Loss()
# pose_mean, pose_std = get_mean_std_necksub(dataloader)
# mean_pose = torch.zeros(1, 64, 104)
# mean_pose = torch.add(mean_pose, pose_mean)
# with torch.no_grad():
#     for ibatch, batch in enumerate(dataloader.test, 1):
#         real_pose = batch['pose/data']
#         real_pose_norm = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)
#         real_neck = real_pose_norm[:, :, :, 0].reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], 2, 1)
#         real_pose_norm = torch.sub(real_pose_norm, real_neck)
#         real_pose_norm = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], -1)
#         real_pose_norm = torch.sub(real_pose_norm, pose_mean)
#         real_pose_norm = torch.div(real_pose_norm, pose_std)

#         rand_pose = randpos_gen(pose_mean, pose_std, batch=real_pose.shape[0], frames=real_pose.shape[1])
        
#         L1_loss_sum_rand += criterion(real_pose_norm, rand_pose)
#         L1_loss_sum_mean += criterion(real_pose_norm, mean_pose)

#         rand_pose_xy = rand_pose.reshape(rand_pose.shape[0], rand_pose.shape[1], 2, -1)
#         rand_gp_torch = torch.flatten(rand_pose_xy, start_dim=0, end_dim=1)
#         rand_gp = rand_gp_torch.numpy()
#         mean_pose_xy = rand_pose.reshape(rand_pose.shape[0], rand_pose.shape[1], 2, -1)
#         mean_gp_torch = torch.flatten(rand_pose_xy, start_dim=0, end_dim=1)
#         mean_gp = rand_gp_torch.numpy()
#         real_pose_xy = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose.shape[1], 2, -1)
#         rp_torch = torch.flatten(real_pose_xy, start_dim=0, end_dim=1)
#         rp = rp_torch.numpy()

#         pck_rand = compute_pck(rand_gp, rp)
#         pck_mean = compute_pck(mean_gp, rp)
#         pck_sum_rand += np.mean(pck_rand, axis=0)
#         pck_sum_mean += np.mean(pck_mean, axis=0)


### testing
pck_sum_rand = 0.
pck_sum_mean = 0.
L1_loss_sum_rand = 0.
L1_loss_sum_mean = 0.
criterion = nn.L1Loss()
pose_mean, pose_std = get_mean_std_necksub(dataloader)
mean_pose = torch.zeros(1, 64, 104)
mean_pose = torch.add(mean_pose, pose_mean)
with torch.no_grad():
    norm_pose_list = []
    for batch in dataloader.test:
        pose = batch['pose/data']
        pose = pose.reshape(pose.shape[0], pose.shape[1], 2, -1)
        neck = pose[:, :, :, 0].reshape(pose.shape[0], pose.shape[1], 2, 1)
        pose = torch.sub(pose, neck)
        pose = pose.reshape(pose.shape[0], pose.shape[1], -1)
        pose = torch.sub(pose, pose_mean)
        pose = torch.div(pose, pose_std)
        norm_pose_list.append(pose)
    for ibatch, batch in enumerate(dataloader.test, 1):
        real_pose = batch['pose/data']
        real_pose_norm = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)
        real_neck = real_pose_norm[:, :, :, 0].reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], 2, 1)
        real_pose_norm = torch.sub(real_pose_norm, real_neck)
        real_pose_norm = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], -1)
        real_pose_norm = torch.sub(real_pose_norm, pose_mean)
        real_pose_norm = torch.div(real_pose_norm, pose_std)

        # rand_pose = randpos_gen(pose_mean, pose_std, batch=real_pose.shape[0], frames=real_pose.shape[1])
        rand_pose = choice(norm_pose_list)

        if ibatch == 1:
            print('rand shape: ', rand_pose.shape)
            # print('rand 1: ', rand_pose[0])
            print('real shape: ', real_pose_norm.shape)
        
        L1_loss_sum_rand += criterion(real_pose_norm, rand_pose)
        # L1_loss_sum_mean += criterion(real_pose_norm, mean_pose)

        rand_pose_xy = rand_pose.reshape(rand_pose.shape[0], rand_pose.shape[1], 2, -1)
        rand_gp_torch = torch.flatten(rand_pose_xy, start_dim=0, end_dim=1)
        rand_gp = rand_gp_torch.numpy()
        # mean_pose_xy = rand_pose.reshape(rand_pose.shape[0], rand_pose.shape[1], 2, -1)
        # mean_gp_torch = torch.flatten(rand_pose_xy, start_dim=0, end_dim=1)
        # mean_gp = rand_gp_torch.numpy()
        real_pose_xy = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], 2, -1)
        rp_torch = torch.flatten(real_pose_xy, start_dim=0, end_dim=1)
        rp = rp_torch.numpy()

        if ibatch == 1:
            print('rand_gp shape: ', rand_gp.shape)
            print('rp shape: ', rp.shape)

        pck_rand = compute_pck(rand_gp, rp)
        # pck_mean = compute_pck(mean_gp, rp)
        pck_sum_rand += np.mean(pck_rand, axis=0)
        # pck_sum_mean += np.mean(pck_mean, axis=0)

        
print('average pck random: ', pck_sum_rand / ibatch)
print('average L1 loss random: ', L1_loss_sum_rand / ibatch)
# print('average pck mean: ', pck_sum_mean / ibatch)
# print('average L1 loss mean: ', L1_loss_sum_mean / ibatch)