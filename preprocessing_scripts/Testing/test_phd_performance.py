from pytorch_implementation.model.net import PhdGifNet
from pytorch_implementation.utils import utils
import torch
from pprint import pprint
import pandas as pd
import os
import numpy as np
from torch.autograd import Variable

MAX_DURATION = 720.0
model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'pytorch_implementation',
    'experiments',
    'phd_gif_model'
)

params_cuda = torch.cuda.is_available()  # use GPU is available
# Set the random seed for reproducible experiments
torch.manual_seed(230)
if params_cuda:
    torch.cuda.manual_seed(230)

# Get the logger
utils.set_logger('test.log')
utils.logging.info("setting up the model")

# Define the model
model = PhdGifNet().cuda() if params_cuda else PhdGifNet()
utils.load_checkpoint_cpu(os.path.join(model_path,'best.pth.tar'), model)
model.eval()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'Data')
VIDEO_FEATURE_DIR = os.path.join(DATA_DIR, 'Test', 'test_video_features')
HISTORY_FEATURE_DIR = os.path.join(DATA_DIR, 'user_history')
CATEGORY_FEATURE_DIR = os.path.join(DATA_DIR, 'text_features')

df = pd.read_csv(os.path.join(DATA_DIR, 'test_dataset.csv'))
sample_test_rows = df[(df['is_last'] == True) & (df['video_duration'] < MAX_DURATION)]

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
        assert 'no category feature available'

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
        user_history_npy = np.loadtxt(data['history_path'], dtype=float)

        video_feature_npy = torch.from_numpy(video_feature_npy.reshape(4096, 1))
        user_history_npy = torch.from_numpy(user_history_npy.reshape(4096, 1))

        video_feature_npy = video_feature_npy.reshape(video_feature_npy.shape[0], -1).float()
        user_history_npy = user_history_npy.reshape(user_history_npy.shape[0], -1).float()

        input_features = torch.cat((video_feature_npy, user_history_npy), dim=0)

        if params_cuda:
            input_features = input_features.cuda(async=True)
            device = torch.device("cuda")

        input_features = Variable(input_features)

        output = model(torch.transpose(input_features, 0, 1))

        predicted = output.data.numpy().flatten()[0]

        test_result_df.iloc[ind, -1] = predicted
    except OSError:
        print(data['feature_path'])

if os.path.exists('phd_gif_result.csv'):
    os.remove('phd_gif_result.csv')
test_result_df.to_csv('phd_gif_result.csv')


##################################################################
# REPORTING THE NUMBERS
##################################################################

if not os.path.exists('phd_gif_result.csv'):
    raise FileNotFoundError('result not available')


df = pd.read_csv('phd_gif_result.csv')

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
PHD GIF 
mAP : {}
nMSD: {}
R@5 : {}
""".format(mAP_score, nMSD_score, r_at_5_score))


"""
PHD GIF 
mAP : 0.03421328585480185
nMSD: 0.5450792516238551
R@5 : 0.15384615384615385
"""
