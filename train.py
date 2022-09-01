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

SPEAKER = 'almaram'
PATS_PATH = '../pats/data'

ROOT_PATH = './save/'+ SPEAKER + '/'
MODEL_PATH_G = ROOT_PATH + 'gen'
MODEL_PATH_D = ROOT_PATH + 'dis'
LOSS_PATH = ROOT_PATH + 'loss.npy'
lr = 10e-4
n_epochs = 6
lambda_d = 1.
lambda_gan = 1.


common_kwargs = dict(path2data = PATS_PATH,
                     speaker = [SPEAKER],
                     modalities = ['pose/data', 'audio/log_mel_512'],
                     fs_new = [15, 15],
                     batch_size = 4,
                     window_hop = 5)

dataloader = Data(**common_kwargs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

# Loss function
motion_regloss = torch.nn.L1Loss()
g_loss = torch.nn.MSELoss()
d_loss1 = torch.nn.MSELoss()
d_loss2 = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Speech2Gesture_G()
discriminator = Speech2Gesture_D(out_channels=64)


if cuda:
    generator.cuda()
    discriminator.cuda()
    motion_regloss.cuda()
    g_loss.cuda()
    d_loss1.cuda()
    d_loss2.cuda()
    


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
g_loss_list = []
d_loss_list = []


pose_mean, pose_std = get_mean_std_necksub(dataloader)
# Normalize the pose
norm_pose_list = []
for batch in dataloader.train:
    pose = batch['pose/data']
    pose = pose.reshape(pose.shape[0], pose.shape[1], 2, -1)
    neck = pose[:, :, :, 0].reshape(pose.shape[0], pose.shape[1], 2, 1)
    pose = torch.sub(pose, neck)
    pose = pose.reshape(pose.shape[0], pose.shape[1], -1)
    pose = torch.sub(pose, pose_mean)
    pose = torch.div(pose, pose_std)
    norm_pose_list.append(pose)

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader.train, 0):

        # Configure input
        audio = batch['audio/log_mel_512']
        audio = audio.to(device)
        audio = audio.type(torch.cuda.FloatTensor)
        real_pose = norm_pose_list[i]
        real_pose = real_pose.to(device)
        real_pose = real_pose.type(torch.cuda.FloatTensor)

        # Adversarial ground truths
        valid = Variable(Tensor(real_pose.size(0), 11).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(real_pose.size(0), 11).fill_(0.0), requires_grad=False)


        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Using audio as generator input
        fake_pose, _ = generator(audio)

        # Generate motions
        real_motion = pos_to_motion(real_pose)
        fake_motion = pos_to_motion(fake_pose)

        # # Generate accelerations
        # real_acceleration = pos_to_motion(real_motion)
        # fake_acceleration = pos_to_motion(fake_motion)

        # discriminator
        fake_d, _ = discriminator(fake_motion)

        # Loss measures generator's ability to fool the discriminator
        G_loss = motion_regloss(real_motion, fake_motion) + lambda_gan * g_loss(fake_d, valid)

        G_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        fake_d, _ = discriminator(fake_motion.detach())
        real_d, _ = discriminator(real_motion)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = d_loss1(real_d, valid)
        fake_loss = d_loss2(fake_d, fake)
        D_loss = real_loss + lambda_d * fake_loss

        D_loss.backward()
        optimizer_D.step()

        if i % 200 == 199:
            print(
                "[Epoch %d/%d] [Batch %d/?] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i+1, D_loss.item(), G_loss.item())
            )
            g_loss_list.append(G_loss.item())
            d_loss_list.append(D_loss.item())

    print('epoch ', epoch, ': ', 'saving generators')
    torch.save(generator.state_dict(), MODEL_PATH_G + 'epoch' + str(epoch))
    print('epoch ', epoch, ': ', 'saving discriminators')
    torch.save(discriminator.state_dict(), MODEL_PATH_D + 'epoch' + str(epoch))
    print('epoch ', epoch, ': ', 'saving losses')
    losses = np.array([g_loss_list, d_loss_list])
    np.save(LOSS_PATH, losses)



