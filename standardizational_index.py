import tqdm
import torch

from pats_master.data import Data

def get_mean_std(dataloader):
    pose_mean_sum = torch.zeros(104)
    pose_squared_sum = torch.zeros(104)

    for batch_num, batch in enumerate(dataloader.train, 1):
        pose = batch['pose/data']
        pose_mean_sum += torch.mean(pose, dim=[0, 1])
        pose_squared_sum += torch.mean(pose ** 2, dim=[0, 1])

    pose_mean = pose_mean_sum / batch_num
    pose_std = (pose_squared_sum / batch_num - pose_mean ** 2) ** 0.5

    # print('mean: ', pose_mean, '\nstd: ', pose_std)
    return pose_mean, pose_std


def get_mean_std_necksub(dataloader):
    pose_mean_sum = torch.zeros(104)
    pose_squared_sum = torch.zeros(104)

    for batch_num, batch in enumerate(dataloader.train, 1):
        pose = batch['pose/data']
        pose = pose.reshape(pose.shape[0], pose.shape[1], 2, -1)
        neck = pose[:, :, :, 0].reshape(pose.shape[0], pose.shape[1], 2, 1)
        pose = torch.sub(pose, neck)
        pose = pose.reshape(pose.shape[0], pose.shape[1], -1)
        pose_mean_sum += torch.mean(pose, dim=[0, 1])
        pose_squared_sum += torch.mean(pose ** 2, dim=[0, 1])

    pose_mean = pose_mean_sum / batch_num
    pose_std = (pose_squared_sum / batch_num - pose_mean ** 2) ** 0.5
    # the mean and std of neck keypoint are both 0, set its std to 1
    pose_std[0] = 1.
    pose_std[52] = 1.

    return pose_mean, pose_std


# common_kwargs = dict(path2data = '../pats/data',
#                      speaker = ['lec_cosmic'],
#                      modalities = ['pose/data'],
#                      fs_new = [15],
#                      batch_size = 4,
#                      window_hop = 5)


# data = Data(**common_kwargs)



