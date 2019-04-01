import pandas as pd
import numpy
import pickle
import os
import glob
import numpy as np
import random


class CreateTrainUserHistory:
    """
    Samples atmost 20 highlight segments
    """

    def __init__(self):
        self.base_data_dir = '/home/adnankhan/PycharmProjects/HighlightDetection/Data/'
        self.video_feature_dir = os.path.join(self.base_data_dir, 'video_features')
        self.user_history_feature_dir = os.path.join(self.base_data_dir, 'user_history')
        self.src_train_file_path = os.path.join(self.base_data_dir, 'train_dataset.csv')
        self.src_test_file_path = os.path.join(self.base_data_dir, 'test_dataset.csv')
        self.max_history = 20
        random.seed(2222)

    def execute_train(self):
        train_df = pd.read_csv(self.src_train_file_path)
        users = train_df['user_id'].unique().tolist()
        for user_id in users:
            user_df = train_df[train_df['user_id'] == user_id]

            user_history_df = user_df[user_df['is_last'] == False]

            history_video_ids = user_history_df['youtubeId'].unique().tolist()

            user_history = np.zeros((4096,))  # verify from the test() in c3d_feature extraction.

            history_highlight_features_files = []

            for v_id in history_video_ids:
                history_highlight_features_files += glob.glob(
                    '{}/{}_{}/highlight/*/*.npy'.format(self.video_feature_dir, v_id, user_id))

            random.shuffle(history_highlight_features_files)

            for path in history_highlight_features_files[:self.max_history]:
                loaded_npy = np.loadtxt(path)
                user_history = np.add(user_history, loaded_npy)

            user_history = user_history / float(self.max_history)

            print('saving history for user {}'.format(user_id))
            np.savetxt(os.path.join(self.user_history_feature_dir, '{}.npy'.format(user_id)), user_history)

        print('completed')

    def execute_test(self):
        test_df = pd.read_csv(self.src_test_file_path)
        users = test_df['user_id'].unique().tolist()
        for user_id in users:
            user_df = test_df[test_df['user_id'] == user_id]

            user_history_df = user_df[user_df['is_last'] == False]

            history_video_ids = user_history_df['youtubeId'].unique().tolist()

            user_history = np.zeros((4096,))  # verify from the test() in c3d_feature extraction.

            history_highlight_features_files = []

            for v_id in history_video_ids:
                history_highlight_features_files += glob.glob(
                    '{}/{}_{}/highlight/*/*.npy'.format(self.video_feature_dir, v_id, user_id))

            random.shuffle(history_highlight_features_files)

            for path in history_highlight_features_files[:self.max_history]:
                loaded_npy = np.loadtxt(path)
                user_history = np.add(user_history, loaded_npy)

            user_history = user_history / float(self.max_history)

            print('saving history for user {}'.format(user_id))
            np.savetxt(os.path.join(self.user_history_feature_dir, '{}.npy'.format(user_id)), user_history)

        print('completed')


if __name__ == '__main__':
    obj = CreateTrainUserHistory()
    # obj.execute_train()
    obj.execute_test()
