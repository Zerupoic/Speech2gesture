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

SPEAKER = 'almaram'
PATS_PATH = '../pats/data'

MODEL_PATH_G = './save/ellen/genepoch3'


common_kwargs = dict(path2data = PATS_PATH,
                     speaker = [SPEAKER],
                     modalities = ['pose/data', 'audio/log_mel_512'],
                     fs_new = [15, 15],
                     batch_size = 4,
                     window_hop = 5)

dataloader = Data(**common_kwargs)



# cuda = True if torch.cuda.is_available() else False

# # Loss function
# motion_regloss = torch.nn.L1Loss()

# generator
generator = Speech2Gesture_G()
generator.load_state_dict(torch.load(MODEL_PATH_G))


### testing
pck_sum = 0.
L1_loss_sum = 0.
criterion = nn.L1Loss()
pose_mean, pose_std = get_mean_std_necksub(dataloader)
with torch.no_grad():
    for ibatch, batch in enumerate(dataloader.test, 1):
        audio = batch['audio/log_mel_512'].type(torch.FloatTensor)
        real_pose = batch['pose/data']
        real_pose_norm = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)
        real_neck = real_pose_norm[:, :, :, 0].reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], 2, 1)
        real_pose_norm = torch.sub(real_pose_norm, real_neck)
        real_pose = real_pose_norm
        real_pose_norm = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], -1)
        real_pose_norm = torch.sub(real_pose_norm, pose_mean)
        real_pose_norm = torch.div(real_pose_norm, pose_std)

        generated_pose_norm, _ = generator(audio)

        generated_pose = torch.mul(generated_pose_norm, pose_std)
        generated_pose = torch.add(generated_pose_norm, pose_mean)
        
        L1_loss_sum += criterion(real_pose_norm, generated_pose_norm)
        # Generate motions
        # real_motion = pos_to_motion(real_pose)
        # generated_motion = pos_to_motion(generated_pose)

        generated_pose_xy = generated_pose.reshape(generated_pose.shape[0], generated_pose.shape[1], 2, -1)
        gp_torch = torch.flatten(generated_pose_xy, start_dim=0, end_dim=1)
        gp = gp_torch.numpy()
        real_pose_xy = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)
        rp_torch = torch.flatten(real_pose_xy, start_dim=0, end_dim=1)
        rp = rp_torch.numpy()
        # if ibatch == 1:
        #     print('gp size: ', gp.shape)
        #     print('rp size: ', rp.shape)
        pck = compute_pck(gp, rp)
        # if ibatch == 1:
        #     print('pck: ', pck.shape)
        pck_sum += np.mean(pck, axis=0)

        # # L1_loss_sum += motion_regloss(real_motion, generated_motion)
        
print('average pck: ', pck_sum / ibatch)
print('average L1 loss: ', L1_loss_sum / ibatch)
