import matplotlib
matplotlib.use("Agg")
import subprocess
import os
import numpy as np
from matplotlib import cm, pyplot as plt
from PIL import Image
from pose_video.consts import BASE_KEYPOINT, RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS, LEFT_HAND_KEYPOINTS, RIGHT_HAND_KEYPOINTS, LINE_WIDTH_CONST


import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch.autograd import Variable

from speech2gesture import *
from pats_master.data import Data
from standardizational_index import get_mean_std, get_mean_std_necksub
from funcs import pos_to_motion

def plot_all_keypoints(img, keypoints, img_width=1280, img_height=720, output=None, title=None, title_x=1, cm=cm.rainbow,
              alpha_img=0.5, alpha_keypoints=None, fig=None, line_width=LINE_WIDTH_CONST):
    if fig is None:
        plt.close("all")
        fig = plt.figure(figsize=(6, 4))

    plt.axis('off')

    if img != None:
        img = Image.open(img)
        img_width, img_height = img.size
    else:
        img = Image.new(mode='RGB', size=(img_width, img_height), color='white')

    plt.imshow(img, alpha=alpha_img)



    keypoints_head = np.array([7, 8, 9])
    plt.plot(keypoints[0][keypoints_head], keypoints[1][keypoints_head], 'o', color='red')

    keypoints_leftarm = np.array([4, 5, 6, 10])
    keypoints_rightarm = np.array([1, 2, 3, 31])
    plt.plot(keypoints[0][keypoints_leftarm], keypoints[1][keypoints_leftarm], 'o', color='blue')
    plt.plot(keypoints[0][keypoints_rightarm], keypoints[1][keypoints_rightarm], 'o', color='purple')

    for i in range(5):
        keypoints_lefthand = np.array(LEFT_HAND_KEYPOINTS(i))
        keypoints_righthand = np.array(LEFT_HAND_KEYPOINTS(i))
        plt.plot(keypoints[0][keypoints_lefthand], keypoints[1][keypoints_lefthand], 'o', color='green')
        plt.plot(keypoints[0][keypoints_righthand], keypoints[1][keypoints_righthand], 'o', color='orange')

    ax = fig.get_axes()[0]
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    # ax.set_ylim(0, img_height)
    if title:
        plt.title(title, x=title_x)

    if output:
        plt.savefig(output)
        plt.close()

def plot_head_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array([7, 8])
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
             color='purple')
    _keypoints = np.array([7, 9])
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
             color='purple')

# COPYED FROM SPEECH2GESTURE ORIGINAL CODE
def plot_body_right_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    # _keypoints = np.array(BASE_KEYPOINT + RIGHT_BODY_KEYPOINTS)
    _keypoints = np.array(RIGHT_BODY_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
             color='gray')


def plot_body_left_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    # _keypoints = np.array(BASE_KEYPOINT + LEFT_BODY_KEYPOINTS)
    _keypoints = np.array(LEFT_BODY_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
             color='blue')


def plot_left_hand_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    for i in range(5):
        _keypoints = np.array(LEFT_HAND_KEYPOINTS(i))
        plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                 color='red')


def plot_right_hand_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    for i in range(5):
        _keypoints = np.array(RIGHT_HAND_KEYPOINTS(i))
        plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                 color='yellow')


def draw_pose(img, keypoints, img_width=1280, img_height=720, output=None, title=None, title_x=1, cm=cm.rainbow,
              alpha_img=0.5, alpha_keypoints=None, fig=None, line_width=LINE_WIDTH_CONST):
    '''
    Note: calling functions must call plt.close() to avoid a memory blowup.
    '''
    if fig is None:
        plt.close("all")
        fig = plt.figure(figsize=(6, 4))

    plt.axis('off')

    if img != None:
        img = Image.open(img)
        img_width, img_height = img.size
    else:
        img = Image.new(mode='RGB', size=(img_width, img_height), color='white')

    plt.imshow(img, alpha=alpha_img)
    plot_head_keypoints(keypoints, alpha_keypoints, line_width)
    plot_body_right_keypoints(keypoints, alpha_keypoints, line_width)
    plot_body_left_keypoints(keypoints, alpha_keypoints, line_width)
    plot_left_hand_keypoints(keypoints, alpha_keypoints, line_width)
    plot_right_hand_keypoints(keypoints, alpha_keypoints, line_width)

    ax = fig.get_axes()[0]
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    # ax.set_ylim(0, img_height)
    if title:
        plt.title(title, x=title_x)

    if output:
        plt.savefig(output)
        plt.close()


def draw_side_by_side_poses(img, keypoints1, keypoints2, output=None, show=True,
                            title="Prediction %s Ground Truth" % (7 * ' '), img_size=(3000, 1000)):
    plt.close("all")
    fig = plt.figure(figsize=(6, 4), dpi=400)
    plt.axis('off')
    if title:
        plt.title(title)
    if img != None:
        img = Image.open(img)
    else:
        img = Image.new(mode='RGB', size=img_size, color='white')

    plt.imshow(img, alpha=0.5)

    for keypoints in [keypoints1, keypoints2]:
        plot_head_keypoints(keypoints)
        plot_body_right_keypoints(keypoints)
        plot_body_left_keypoints(keypoints)
        plot_left_hand_keypoints(keypoints)
        plot_right_hand_keypoints(keypoints)

    if show:
        plt.show()
    if output is not None:
        plt.savefig(output)
    return fig


def save_side_by_side_video(temp_folder, keypoints1, keypoints2, output_fn, delete_tmp=True):
    if not (os.path.exists(temp_folder)):
        os.makedirs(temp_folder)

    if not (os.path.exists(os.path.dirname(output_fn))):
        os.makedirs(temp_folder)

    output_fn_pattern = os.path.join(temp_folder, '%04d.jpg')

    diff = len(keypoints2) - len(keypoints1)
    if diff > 0:
        conditioned_keypoints = keypoints2[:diff]
        keypoints2 = keypoints2[diff:]
        for i in range(len(conditioned_keypoints)):
            draw_pose(img=None, keypoints=conditioned_keypoints[i], img_width=3000, img_height=1000,
                      output=output_fn_pattern % i, title="Input", title_x=0.63)

    for j in range(len(keypoints1)):
        draw_side_by_side_poses(None, keypoints1[j], keypoints2[j], output=output_fn_pattern % (j + diff), show=False)
        plt.close()

    create_mute_video_from_images(output_fn, temp_folder)
    if delete_tmp:
        subprocess.call('rm -R "%s"' % (temp_folder), shell=True)


def create_mute_video_from_images(output_fn, temp_folder):
    '''
    :param output_fn: output video file name
    :param temp_folder: contains images in the format 0001.jpg, 0002.jpg....
    :return:
    '''
    subprocess.call('ffmpeg -loglevel panic -r 30000/2002 -f image2 -i "%s" -r 30000/1001 "%s" -y' % (
        os.path.join(temp_folder, '%04d.jpg'), output_fn), shell=True)


def save_video_from_audio_video(audio_input_path, input_video_path, output_video_path):
    subprocess.call(
        'ffmpeg -loglevel panic -i "%s" -i "%s" -strict -2 "%s" -y' % (
        audio_input_path, input_video_path, output_video_path),
        shell=True)



SPEAKER = 'almaram'
PATS_PATH = '../pats/data'

ROOT_PATH = './save/'+ SPEAKER + '/'
MODEL_PATH_G = ROOT_PATH + 'genepoch5'


common_kwargs = dict(path2data = PATS_PATH,
                     speaker = [SPEAKER],
                     modalities = ['pose/data', 'audio/log_mel_512'],
                     fs_new = [15, 15],
                     batch_size = 4,
                     window_hop = 5)

dataloader = Data(**common_kwargs)

# generator
generator = Speech2Gesture_G()
generator.load_state_dict(torch.load(MODEL_PATH_G))

for batch in dataloader.test:
    break

pose_mean, pose_std = get_mean_std_necksub(dataloader)
with torch.no_grad():
    audio = batch['audio/log_mel_512'].type(torch.FloatTensor)
    real_pose = batch['pose/data']
    real_pose_norm = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)
    ori_pose = real_pose_norm
    real_neck = real_pose_norm[:, :, :, 0].reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], 2, -1)
    real_pose_norm = torch.sub(real_pose_norm, real_neck)
    real_pose = real_pose_norm
    real_pose_norm = real_pose_norm.reshape(real_pose_norm.shape[0], real_pose_norm.shape[1], -1)
    real_pose_norm = torch.sub(real_pose_norm, pose_mean)
    real_pose_norm = torch.div(real_pose_norm, pose_std)

    generated_pose_norm, _ = generator(audio)

    generated_pose = torch.mul(generated_pose_norm, pose_std)
    generated_pose = torch.add(generated_pose_norm, pose_mean)

    rp_torch = real_pose.reshape(real_pose.shape[0], real_pose.shape[1], 2, -1)[1]
    # rp_torch = torch.flatten(rp_torch, start_dim=0, end_dim=1)
    rp = rp_torch.numpy()
    gp_torch = generated_pose.reshape(generated_pose.shape[0], generated_pose.shape[1], 2, -1)[1]
    # gp_torch = torch.flatten(gp_torch, start_dim=0, end_dim=1)
    gp = gp_torch.numpy()

    # op = ori_pose[0].numpy()

# print('rp shape: ', rp.shape)
# print('rp[31]: ', rp[31])
# print('op shape: ', op.shape)
# print('op[31]: ', op[31])
# print('gp[31]: ', gp[31])
# print('rp[31] - gp[31]', rp[31] - gp[31])

# centre1 = np.array([[700], [360]])
# centre2 = np.array([[2100], [360]])
size = np.array([[3, 0], [0, -3]])
# gp = np.matmul(size, gp) + centre1
# rp = np.matmul(size, rp) + centre2
rp = np.matmul(size, rp) - np.array([[1500], [0]])
gp = np.matmul(size, gp)

# plot_all_keypoints(img=None, keypoints=-rp[31], img_width=1280, img_height=720, output='./save/drawall_rp.jpg', title='real pose', title_x=1, cm=cm.rainbow, alpha_img=0.5)
# plot_all_keypoints(img=None, keypoints=-gp[31], img_width=1280, img_height=720, output='./save/drawall_gp.jpg', title='generated pose', title_x=1, cm=cm.rainbow, alpha_img=0.5)
# draw_pose(img=None, keypoints=-rp[31], img_width=1280, img_height=720, output='./save/output.jpg', title='real pose', title_x=1, cm=cm.rainbow, alpha_img=0.5, alpha_keypoints=None, fig=None, line_width=LINE_WIDTH_CONST)

save_side_by_side_video('./save/temp', -rp, -gp, './videos', delete_tmp=False)
