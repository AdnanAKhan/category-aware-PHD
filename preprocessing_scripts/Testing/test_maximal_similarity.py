from pytorch_implementation.model.net import Net
from pytorch_implementation.utils import utils
import torch
from pprint import pprint
import pandas as pd
import os
import numpy as np
from torch.autograd import Variable


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'Data')
VIDEO_FEATURE_DIR = os.path.join(DATA_DIR, 'Test', 'test_video_features')
HISTORY_FEATURE_DIR = os.path.join(DATA_DIR, 'user_history')
CATEGORY_FEATURE_DIR = os.path.join(DATA_DIR, 'text_features')

df = pd.read_csv(os.path.join(DATA_DIR, 'test_dataset.csv'))
sample_test_rows = df[(df['is_last'] == True) & (df['video_duration'] < 720.0)]

test_result_df = pd.DataFrame(
    columns=['youtubeId', 'user_id', 'feature_path', 'history_path', 'category_path', 'ground_label', 'predicted'])

for ind, data in sample_test_rows.iterrows():
    # user history path
    if os.path.exists(os.path.join(HISTORY_FEATURE_DIR, '{}.npy'.format(data['user_id']))):
        history_path = os.path.join(HISTORY_FEATURE_DIR, '{}.npy'.format(data['user_id']))
    else:
        assert 'no hisory available'

    # text feature path
    if os.path.exists(os.path.join(CATEGORY_FEATURE_DIR, '{}.npy'.format(data['youtubeId']))):
        category_path = os.path.join(CATEGORY_FEATURE_DIR, '{}.npy'.format(data['youtubeId']))
    else:
        assert 'no hisory available'

    # video feature paths
    video_feature_path = os.path.join(VIDEO_FEATURE_DIR, '{}_{}'.format(data['youtubeId'], data['user_id']))
    for folder in os.listdir(video_feature_path):
        label = folder.split('_')[-1]
        feature_path = os.path.join(video_feature_path, folder, 'feature.npy')
        test_result_df = test_result_df.append({
            'youtubeId': data['youtubeId'],
            'user_id': data['user_id'],
            'feature_path': feature_path,
            'history_path': history_path,
            'category_path': category_path,
            'ground_label': label,
            'predicted': 1,
        }, ignore_index=True)

for ind, data in test_result_df.iterrows():
    try:
        video_feature_npy = np.loadtxt(data['feature_path'], dtype=float)
        text_feature_npy = np.loadtxt(data['category_path'], dtype=float)
        user_history_npy = np.loadtxt(data['history_path'], dtype=float)

        video_feature_npy = torch.from_numpy(video_feature_npy.reshape(4096, 1))
        text_feature_npy = torch.from_numpy(text_feature_npy.reshape(100, 1))
        user_history_npy = torch.from_numpy(user_history_npy.reshape(4096, 1))

        video_feature_npy = video_feature_npy.reshape(video_feature_npy.shape[0], -1).float()
        text_feature_npy = text_feature_npy.reshape(text_feature_npy.shape[0], -1).float()
        user_history_npy = user_history_npy.reshape(user_history_npy.shape[0], -1).float()

        video_feature = Variable(video_feature_npy)
        history_feature = Variable(user_history_npy)

        cosine_similarity_score = torch.nn.functional.cosine_similarity(video_feature, history_feature, dim=0, eps=1e-8)
        test_result_df.iloc[ind, -1] = cosine_similarity_score.data.numpy()[0]
    except OSError:
        print(data['feature_path'])

if os.path.exists('maximal_similarity.csv'):
    os.remove('maximal_similarity.csv')
test_result_df.to_csv('maximal_similarity.csv')

##################################################################
# REPORTING THE NUMBERS
##################################################################

if not os.path.exists('maximal_similarity.csv'):
    raise FileNotFoundError('result not available')


df = pd.read_csv('maximal_similarity.csv')

y_true = df['ground_label'].values
y_predict = df['predicted'].values

assert isinstance(y_true, np.ndarray), "ground truth needed to be np.array"
assert isinstance(y_predict, np.ndarray), "predicted label needed to be np.array"

from preprocessing_scripts.Testing.evaluation import get_ap
mAP_score = get_ap(y_true, y_predict)

from preprocessing_scripts.Testing.evaluation import meaningful_summary_duration

nMSD=[]
for grp, data in df.groupby(['youtubeId', 'user_id']):
    highlight_ground_truths = [data['ground_label'].values]
    y_predict = data['predicted'].values
    if not highlight_ground_truths[0].sum() == 0:
        nMSD.append(meaningful_summary_duration(highlight_ground_truths, y_predict))

nMSD_score = np.array(nMSD).mean()

r_at_5=[]
for grp, data in df.groupby(['youtubeId', 'user_id']):
    highlight_ground_truths = [data['ground_label'].values]
    # if not highlight_ground_truths[0].sum() == 0:
    d= data.copy()
    d.sort_values(['predicted'], ascending=False, inplace=True)
    r_at_5.append(d[:5]['ground_label'].sum())

r_at_5_score=np.array(r_at_5).mean(axis=0)


print("""
Maximal Similarity
mAP : {}
nMSD: {}
R@5 : {}
""".format(mAP_score, nMSD_score, r_at_5_score))

""""
Maximal Similarity
mAP : 0.03283414022474966
nMSD: 0.5574415635875399
R@5 : 0.20279720279720279
"""

