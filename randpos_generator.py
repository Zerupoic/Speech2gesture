import numpy as np
import torch
import tqdm

from torch.autograd import Variable

from speech2gesture import *
from pats_master.data import Data
from standardizational_index import get_mean_std, get_mean_std_necksub
from funcs import pos_to_motion
from evaluation import compute_pck, compute_pck_radius


def randpos_gen(mean, std, batch=4, frames=64, keypoints=104):
    rand_pose = torch.randn((batch, frames, keypoints))
    rand_pose = torch.mul(rand_pose, std)
    rand_pose = torch.add(rand_pose, mean)
    return rand_pose



# mean = torch.ones(104)
# std = torch.ones(104)
# print(torch.mean(randpos_gen(mean, std)))

