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
  pre_bbox = scod_sample['pre_frame']['bbox']
  pnr_bbox = scod_sample['pnr_frame']['bbox']
  post_bbox = scod_sample['post_frame']['bbox']
  return (pre_img, pre_bbox), (pnr_img, pnr_bbox), (post_img, post_bbox)

def plot_img(ax, img, bboxes=None, sample_idx=0, state='', title=''):
  ax.imshow(img)
  print(f"Sample {sample_idx} {state} frame has objects:")
  for bbox in bboxes:
    print(f"     {bbox['object_type']} with structured noun: {bbox['structured_noun']}")
    bbox = bbox['bbox']
    rect = patches.Rectangle((bbox['x'], bbox['y']), bbox['width'], bbox['height'], linewidth=1, facecolor='none', edgecolor='r')
    ax.add_patch(rect)
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)
  ax.set_title(title)
  return
#scod_idx = np.random.randint(0, scod_num_clips - 1)
#hands_idx = np.random.randint(0, hands_num_clips - 1)

sample_idx = 0
(pre_img, pre_bbox), (pnr_img, pnr_bbox), (post_img, post_bbox) = get_frames_and_bboxes(sample_idx)
fig, ax = plt.subplots(3, 3)
fig.set_figheight(5)
fig.set_figwidth(8)

plot_img(ax[0,0], pre_img, bboxes=pre_bbox, sample_idx=sample_idx, state='pre', title='Pre Frame' + str(sample_idx))
plot_img(ax[0,1], pnr_img, bboxes=pnr_bbox, sample_idx=sample_idx, state='pnr', title='PNR Frame' + str(sample_idx))
plot_img(ax[0,2], post_img, bboxes=post_bbox, sample_idx=sample_idx, state='post', title='Post Frame' + str(sample_idx))

sample_idx = 7584
plot_img(ax[1,0], pre_img, bboxes=pre_bbox, sample_idx=sample_idx, state='pre', title='Pre Frame' + str(sample_idx))
plot_img(ax[1,1], pnr_img, bboxes=pnr_bbox, sample_idx=sample_idx, state='pnr', title='PNR Frame' + str(sample_idx))
plot_img(ax[1,2], post_img, bboxes=post_bbox, sample_idx=sample_idx, state='post', title='Post Frame' + str(sample_idx))

sample_idx = 8492
plot_img(ax[2,0], pre_img, bboxes=pre_bbox, sample_idx=sample_idx, state='pre', title='Pre Frame' + str(sample_idx))
plot_img(ax[2,1], pnr_img, bboxes=pnr_bbox, sample_idx=sample_idx, state='pnr', title='PNR Frame' + str(sample_idx))
plot_img(ax[2,2], post_img, bboxes=post_bbox, sample_idx=sample_idx, state='post', title='Post Frame' + str(sample_idx))

plt.savefig("pnr_loc")
