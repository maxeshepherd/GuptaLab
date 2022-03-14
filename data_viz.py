import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
import skvideo.io
import sys

PATH_TO_ANN = "/home/saurabhg/ego4d-all/ego4d-data/v1/annotations"
PATH_TO_VIDEOS = "/home/saurabhg/ego4d-all/ego4d-data/v1/clips"
SCOD_FILE = os.path.join(PATH_TO_ANN, "fho_scod_val.json")
HANDS_FILE = os.path.join(PATH_TO_ANN, "fho_hands_val.json")
  
with open(SCOD_FILE, 'r') as f:
  scod_data = json.load(f)


with open(HANDS_FILE, 'r') as f:
  hands_data = json.load(f)

scod_num_clips = len(scod_data['clips'])
hands_num_clips = len(hands_data['clips'])
print(f"State Change Object detection has {len(scod_data['clips'])} clips")
print(f"PNR temporal localization has {len(hands_data['clips'])} clips")

def get_frames(idx):
  scod_sample = scod_data['clips'][idx]
  clip_uid = scod_sample['clip_uid']
  post_frame = scod_sample['post_frame']['clip_frame_number']
  pre_frame = scod_sample['pre_frame']['clip_frame_number']
  pnr_frame = scod_sample['pnr_frame']['clip_frame_number']
  video_path = os.path.join(PATH_TO_VIDEOS, clip_uid+".mp4")
  videodata = skvideo.io.vread(video_path)
  post_img = videodata[post_frame]
  pre_img = videodata[pre_frame]
  pnr_img = videodata[pnr_frame]
  return pre_img, pnr_img, post_img

#scod_idx = np.random.randint(0, scod_num_clips - 1)
#hands_idx = np.random.randint(0, hands_num_clips - 1)

scod_idx = 0
hands_idx = 0

pre_img, pnr_img, post_img = get_frames(0)
fig, ax = plt.subplots(3, 3)
fig.set_figheight(5)
fig.set_figwidth(8)
ax[0,0].imshow(pre_img)
ax[0,0].axes.xaxis.set_visible(False)
ax[0,0].axes.yaxis.set_visible(False)
ax[0,0].set_title("Pre Frame")
ax[0,1].imshow(pnr_img)
ax[0,1].axes.xaxis.set_visible(False)
ax[0,1].axes.yaxis.set_visible(False)
ax[0,1].set_title("PNR Frame")
ax[0,2].imshow(post_img)
ax[0,2].axes.xaxis.set_visible(False)
ax[0,2].axes.yaxis.set_visible(False)
ax[0,2].set_title("Post Frame")

#pre_img, pnr_img, post_img = get_frames(1)
ax[1,0].imshow(pre_img)
ax[1,0].axes.xaxis.set_visible(False)
ax[1,0].axes.yaxis.set_visible(False)
ax[1,0].set_title("Pre Frame")
ax[1,1].imshow(pnr_img)
ax[1,1].axes.xaxis.set_visible(False)
ax[1,1].axes.yaxis.set_visible(False)
ax[1,1].set_title("PNR Frame")
ax[1,2].imshow(post_img)
ax[1,2].axes.xaxis.set_visible(False)
ax[1,2].axes.yaxis.set_visible(False)
ax[1,2].set_title("Post Frame")

#pre_img, pnr_img, post_img = get_frames(2)
ax[2,0].imshow(pre_img)
ax[2,0].axes.xaxis.set_visible(False)
ax[2,0].axes.yaxis.set_visible(False)
ax[2,0].set_title("Pre Frame")
ax[2,1].imshow(pnr_img)
ax[2,1].axes.xaxis.set_visible(False)
ax[2,1].axes.yaxis.set_visible(False)
ax[2,1].set_title("PNR Frame")
ax[2,2].imshow(post_img)
ax[2,2].axes.xaxis.set_visible(False)
ax[2,2].axes.yaxis.set_visible(False)
ax[2,2].set_title("Post Frame")
plt.savefig("pnr_loc")
