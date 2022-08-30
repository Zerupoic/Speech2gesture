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


SAVE_PATH = './save/testdata_sample/almaram_pose.npy'


common_kwargs = dict(path2data = '../pats/data',
                     speaker = ['almaram'],
                     modalities = ['pose/data'],
                     fs_new = [15, 15],
                     batch_size = 4,
                     window_hop = 5)

dataloader = Data(**common_kwargs)

# pose_mean, pose_std = get_mean_std(dataloader)
# pose_mean_necksub, pose_std_necksub = get_mean_std_necksub(dataloader)

# print('pose mean: ', pose_mean)
# print('pose std: ', pose_std)
# print('pose mean necksub: ', pose_mean_necksub)
# print('pose std necksub: ', pose_std_necksub)

pose_mean, pose_std = get_mean_std_necksub(dataloader)
# pose_list = []
for ibatch, batch in enumerate(dataloader.test, 0):
    if ibatch == 1:
        break
# for batch in dataloader.test[:10]:
    pose = batch['pose/data']
    pose = pose.reshape(pose.shape[0], pose.shape[1], 2, -1)
    neck = pose[:, :, :, 0].reshape(pose.shape[0], pose.shape[1], 2, 1)
    pose = torch.sub(pose, neck)
    pose = pose.reshape(pose.shape[0], pose.shape[1], -1)
    pose = torch.sub(pose, pose_mean)
    pose = torch.div(pose, pose_std)
    # pose_list.extend(list(pose))
    sample_pose = list(pose[0:1, :, :])
pose_np = np.array(sample_pose)
np.save(SAVE_PATH, pose_np)






# pose_list = np.load(SAVE_PATH)
# print(pose_list.shape)
# print(pose_list[5][10])






# MODEL_PATH_G = './save/gen_almaram_norm_correctlam'
# SAVE_PATH = './save/testdata_sample/almaram_audio.npy'
# SAVE_PATH2 = './save/testdata_sample/almaram_gen_pose.npy'


# common_kwargs = dict(path2data = '../pats/data',
#                      speaker = ['almaram'],
#                      modalities = ['audio/log_mel_512'],
#                      fs_new = [15, 15],
#                      batch_size = 4,
#                      window_hop = 5)

# dataloader = Data(**common_kwargs)

# # generator
# generator = Speech2Gesture_G()
# generator.load_state_dict(torch.load(MODEL_PATH_G))

# audio_list = []
# gen_pose_list = []
# for ibatch, batch in enumerate(dataloader.test, 0):
#     if ibatch == 10:
#         break
#     audio = batch['audio/log_mel_512']
#     gen_pose, _ = generator(audio)
#     audio_list.extend(list(audio))
#     gen_pose_list.extend(list(gen_pose))
# audio_np = np.array(audio_list)
# gen_pose_np = np.array(gen_pose_list)
# np.save(SAVE_PATH, audio_np)
# np.save(SAVE_PATH2, gen_pose_np)
    
