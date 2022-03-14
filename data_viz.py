import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def get_frames_and_bboxes(idx):
  scod_sample = scod_data['clips'][idx]
  clip_uid = scod_sample['clip_uid']
  pre_frame = scod_sample['pre_frame']['clip_frame_number']
  pnr_frame = scod_sample['pnr_frame']['clip_frame_number']
  post_frame = scod_sample['post_frame']['clip_frame_number']
  video_path = os.path.join(PATH_TO_VIDEOS, clip_uid+".mp4")
  videodata = skvideo.io.vread(video_path)
  post_img = videodata[post_frame]
  pre_img = videodata[pre_frame]
  pnr_img = videodata[pnr_frame]
  pre_bbox = scod_sample['pre_frame']['bbox'][0]['bbox']
  pnr_bbox = scod_sample['pnr_frame']['bbox'][0]['bbox']
  post_bbox = scod_sample['post_frame']['bbox'][0]['bbox']
  return (pre_img, pre_bbox), (pnr_img, pnr_bbox), (post_img, post_bbox)
#scod_idx = np.random.randint(0, scod_num_clips - 1)
#hands_idx = np.random.randint(0, hands_num_clips - 1)

scod_idx = 0
hands_idx = 0
(pre_img, pre_bbox), (pnr_img, pnr_bbox), (post_img, post_bbox) = get_frames_and_bboxes(0)
fig, ax = plt.subplots(3, 3)
fig.set_figheight(5)
fig.set_figwidth(8)
ax[0,0].imshow(pre_img)
rect = patches.Rectangle((pre_bbox['x'], pre_bbox['y']), pre_bbox['width'], pre_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[0,0].add_patch(rect)
ax[0,0].axes.xaxis.set_visible(False)
ax[0,0].axes.yaxis.set_visible(False)
ax[0,0].set_title("Pre Frame")
ax[0,1].imshow(pnr_img)
rect = patches.Rectangle((pnr_bbox['x'], pnr_bbox['y']), pnr_bbox['width'], pnr_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[0,1].add_patch(rect)
ax[0,1].axes.xaxis.set_visible(False)
ax[0,1].axes.yaxis.set_visible(False)
ax[0,1].set_title("PNR Frame")
ax[0,2].imshow(post_img)
rect = patches.Rectangle((post_bbox['x'], post_bbox['y']), post_bbox['width'], post_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[0,2].add_patch(rect)
ax[0,2].axes.xaxis.set_visible(False)
ax[0,2].axes.yaxis.set_visible(False)
ax[0,2].set_title("Post Frame")

(pre_img, pre_bbox), (pnr_img, pnr_bbox), (post_img, post_bbox) = get_frames_and_bboxes(790)
ax[1,0].imshow(pre_img)
rect = patches.Rectangle((pre_bbox['x'], pre_bbox['y']), pre_bbox['width'], pre_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[1,0].add_patch(rect)
ax[1,0].axes.xaxis.set_visible(False)
ax[1,0].axes.yaxis.set_visible(False)
ax[1,0].set_title("Pre Frame")
ax[1,1].imshow(pnr_img)
rect = patches.Rectangle((pnr_bbox['x'], pnr_bbox['y']), pnr_bbox['width'], pnr_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[1,1].add_patch(rect)
ax[1,1].axes.xaxis.set_visible(False)
ax[1,1].axes.yaxis.set_visible(False)
ax[1,1].set_title("PNR Frame")
ax[1,2].imshow(post_img)
rect = patches.Rectangle((post_bbox['x'], post_bbox['y']), post_bbox['width'], post_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[1,2].add_patch(rect)
ax[1,2].axes.xaxis.set_visible(False)
ax[1,2].axes.yaxis.set_visible(False)
ax[1,2].set_title("Post Frame")

(pre_img, pre_bbox), (pnr_img, pnr_bbox), (post_img, post_bbox) = get_frames_and_bboxes(4499)
fig, ax = plt.subplots(3, 3)
fig.set_figheight(5)
fig.set_figwidth(8)
ax[2,0].imshow(pre_img)
rect = patches.Rectangle((pre_bbox['x'], pre_bbox['y']), pre_bbox['width'], pre_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[2,0].add_patch(rect)
ax[2,0].axes.xaxis.set_visible(False)
ax[2,0].axes.yaxis.set_visible(False)
ax[2,0].set_title("Pre Frame")
ax[2,1].imshow(pnr_img)
rect = patches.Rectangle((pnr_bbox['x'], pnr_bbox['y']), pnr_bbox['width'], pnr_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[2,1].add_patch(rect)
ax[2,1].axes.xaxis.set_visible(False)
ax[2,1].axes.yaxis.set_visible(False)
ax[2,1].set_title("PNR Frame")
ax[2,2].imshow(post_img)
rect = patches.Rectangle((post_bbox['x'], post_bbox['y']), post_bbox['width'], post_bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
ax[2,2].add_patch(rect)
ax[2,2].axes.xaxis.set_visible(False)
ax[2,2].axes.yaxis.set_visible(False)
ax[2,2].set_title("Post Frame")
plt.savefig("pnr_loc")
