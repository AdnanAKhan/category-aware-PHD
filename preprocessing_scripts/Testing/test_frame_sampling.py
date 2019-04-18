import os
import pandas as pd
import numpy as np
import pickle
import subprocess
from copy import deepcopy
from pprint import pprint


def get_segment_label(clip_start, clip_end, highl_start, highl_end):
    if highl_start > clip_end and highl_end > clip_end: # 1
        over_lap = 0.0
    elif highl_start < clip_start and highl_end < clip_start: # 2
        over_lap = 0.0
    elif highl_start >= clip_start and highl_end <= clip_end: # 3
        over_lap = 1.0
    elif highl_start <=clip_start and highl_end >= clip_end: # 6
        over_lap = 1.0
    elif highl_start <= clip_start <=highl_end and highl_end <=clip_end: # 4
        over_lap = np.abs(clip_start-highl_end)/5.0
    elif clip_start<= highl_start <=clip_end and highl_end > clip_end: # 5
        over_lap = np.abs(highl_start- clip_end) / 5.0
    else:
        over_lap = 0.0
    return over_lap


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Data')
YOUTUBE_VIDEO_DIR = os.path.join('/media/adnankhan/94CAA53DCAA51C8C/video')
YOUTUBE_DL_VIDEO_DIR = os.path.join('/media/adnankhan/94CAA53DCAA51C8C/youtube_dl')

SEGMENT_DURATION = 5.0

df = pd.read_csv(os.path.join(DATA_DIR, 'test_dataset.csv'))
sample_test_rows = df[(df['is_last'] == True) & (df['video_duration'] < 720.0)]

user_id = sample_test_rows['user_id'].unique().tolist()
youtubeId = sample_test_rows['youtubeId'].unique().tolist()
keywords = sample_test_rows['keyword'].unique().tolist()

DEST_DIR = os.path.join(DATA_DIR, 'Test', 'test_videos')
for idx, data in sample_test_rows.iterrows():
    dest_test_video_location = os.path.join(DEST_DIR, "{}_{}".format(data['youtubeId'], data['user_id']))
    if not os.path.exists(dest_test_video_location):
        os.makedirs(dest_test_video_location)

    if os.path.exists(os.path.join(YOUTUBE_VIDEO_DIR, '{}.mp4'.format(data['youtubeId']))):
        raw_video_path = os.path.join(YOUTUBE_VIDEO_DIR, '{}.mp4'.format(data['youtubeId']))
    elif os.path.exists(os.path.join(YOUTUBE_DL_VIDEO_DIR, '{}.mp4'.format(data['youtubeId']))):
        raw_video_path = os.path.join(YOUTUBE_DL_VIDEO_DIR, '{}.mp4'.format(data['youtubeId']))
    else:
        print('not available')

    total_duration = data['video_duration']
    highlight_start = data['start']
    highlight_end = data['start'] + data['duration']

    video_segment_info = []
    for i in np.arange(0.0, total_duration, SEGMENT_DURATION):
        if i + 5.0 > total_duration:
            seg_start = total_duration - SEGMENT_DURATION + 0.001
            seg_end = total_duration
            data = {
                'start': seg_start,
                'end': seg_end,
                'label': 1 if get_segment_label(seg_start, seg_end, highlight_start, highlight_end) > 0.66 else 0
            }
        else:
            seg_start = i
            seg_end = i + SEGMENT_DURATION - 0.001
            data = {
                'start': seg_start,
                'end': seg_end,
                'label': 1 if get_segment_label(seg_start, seg_end, highlight_start, highlight_end) > 0.66 else 0
            }

        video_segment_info.append(deepcopy(data))

    for ind, item in enumerate(video_segment_info):
        dest_test_segment_store_path = os.path.join(dest_test_video_location, 'segment_{}_{}'.format(ind+1, item['label']))
        if not os.path.exists(dest_test_segment_store_path):
            os.makedirs(dest_test_segment_store_path)

            output_format = os.path.join(dest_test_segment_store_path, 'frame%02d.jpg')

            subprocess.call(['ffmpeg',
                             '-v',
                             '0',
                             '-i',
                             raw_video_path,
                             '-vf',
                             "select='between(t,{},{})'".format(item['start'], item['end']),
                             # '-vf',
                             # "fps={}".format('1'),
                             '-s',
                             '{}x{}'.format(320, 320),
                             "-vframes",
                             '16',
                             '-vsync',
                             '0',
                             output_format,
                             '-hide_banner'
                             ])
