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

#scod_idx = np.random.randint(0, scod_num_clips - 1)
#hands_idx = np.random.randint(0, hands_num_clips - 1)

scod_idx = 0
hands_idx = 0

scod_sample = scod_data['clips'][scod_idx]
hands_sample = hands_data['clips'][hands_idx]

clip_id = hands_sample['clip_id']
clip_uid = hands_sample['clip_uid']
video_uid = hands_sample['video_uid']
hands_frames = hands_sample['frames'][0]
action_start_sec = hands_frames['action_start_sec']
action_end_sec = hands_frames['action_end_sec']
action_start_frame = hands_frames['action_start_frame']
action_end_frame = hands_frames['action_end_frame']
action_clip_start_sec = hands_frames['action_clip_start_sec']
action_clip_end_sec = hands_frames['action_clip_end_sec']
action_clip_start_frame = hands_frames['action_clip_start_frame']
action_clip_end_frame = hands_frames['action_clip_end_frame']
pre_45 = hands_frames['pre_45']['clip_frame']
pre_30 = hands_frames['pre_30']['clip_frame']
pre_15 = hands_frames['pre_15']['clip_frame']
post_frame = hands_frames['post_frame']['clip_frame']
pre_frame = hands_frames['pre_frame']['clip_frame']
pnr_frame = hands_frames['pnr_frame']['clip_frame']

print(f"Clip is from video {video_uid} from clip uid {clip_uid} and some important values are: {hands_frames}")
VIDEO_PATH = os.path.join(PATH_TO_VIDEOS, clip_uid+".mp4")
videodata = skvideo.io.vread(VIDEO_PATH)

post_img = videodata[post_frame]
pre_img = videodata[pre_frame]
pnr_img = videodata[pnr_frame]

fig, ax = plt.subplots(1, 3)
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
plt.savefig("pnr_loc")
