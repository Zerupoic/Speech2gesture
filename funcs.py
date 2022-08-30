import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from speech2gesture import *

def pos_to_motion(pose_batch):
    # shape = pose_batch.shape()
    # reshaped = pose.reshape(shape[0], shape[1], 2, -1)
    # diff = pose_batch[:, 1:] - pose_batch[:, :-1]
    diff = torch.diff(pose_batch, n=1, dim=1)
    return diff



