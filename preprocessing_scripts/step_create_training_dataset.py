import pandas as pd
import numpy
import pickle
import os
import glob
import csv


class CreateTrainingDatasetFile:

    def __init__(self):
        self.base_data_dir = '/home/adnankhan/PycharmProjects/HighlightDetection/Data/'
        self.user_history_feature_dir = os.path.join(self.base_data_dir, 'user_history')
        self.video_feature_dir = os.path.join(self.base_data_dir, 'video_features')
        # self.glove_feature_dir = os.path.join(self.base_data_dir, 'glove_features')
        self.glove_feature_dir = os.path.join('glove_features')
        self.src_train_file_path = os.path.join(self.base_data_dir, 'train_dataset.csv')
        self.dest_train_file_path = os.path.join(self.base_data_dir, 'train_dataset_pytorch.csv')

        self.src_test_file_path = os.path.join(self.base_data_dir, 'test_dataset.csv')
        self.dest_test_file_path = os.path.join(self.base_data_dir, 'test_dataset_pytorch.csv')

    def execute_train(self):
        train_df = pd.read_csv(self.src_train_file_path)
        users = train_df['user_id'].unique().tolist()

        with open(self.dest_train_file_path, 'w') as csvfile:
            str_row = "{},{},{},{}{}".format('highlight',
                                             'non_highlight',
                                             'user_history',
                                             'word_vector',
                                             os.linesep)
            csvfile.write(str_row)
            for user_id in users:
                user_df = train_df[train_df['user_id'] == user_id]

                train_data_df = user_df[user_df['is_last'] == True]

                train_data_video_id = train_data_df['youtubeId'].unique().tolist()

                if len(train_data_video_id) > 0:

                    train_highlight_features_paths = []

                    for v_id in train_data_video_id:
                        train_highlight_features_paths += glob.glob(
                            '{}/{}_{}/highlight/*/*.npy'.format(self.video_feature_dir, v_id, user_id))

                    train_non_highlight_features_paths = []
                    for v_id in train_data_video_id:
                        train_non_highlight_features_paths += glob.glob(
                            '{}/{}_{}/non_highlight/*/*.npy'.format(self.video_feature_dir, v_id, user_id))

                    user_history_feature_path = os.path.join(self.user_history_feature_dir, '{}.npy'.format(user_id))
                    word_feature_representation_path = os.path.join(self.glove_feature_dir,
                                                                    '{}.p'.format(train_data_video_id[0]))

                    for positive in train_highlight_features_paths:
                        for negative in train_non_highlight_features_paths:
                            str_row = "{},{},{},{} {}".format(positive.lstrip(self.base_data_dir),
                                                              negative.lstrip(self.base_data_dir),
                                                              user_history_feature_path.lstrip(self.base_data_dir),
                                                              word_feature_representation_path,
                                                              os.linesep)
                            csvfile.write(str_row)

    def execute_test(self):
        test_df = pd.read_csv(self.src_test_file_path)
        users = test_df['user_id'].unique().tolist()

        with open(self.dest_test_file_path, 'w') as csvfile:
            str_row = "{},{},{},{}{}".format('highlight',
                                             'non_highlight',
                                             'user_history',
                                             'word_vector',
                                             os.linesep)
            csvfile.write(str_row)
            for user_id in users:
                user_df = test_df[test_df['user_id'] == user_id]

                test_data_df = user_df[user_df['is_last'] == True]

                test_data_video_id = test_data_df['youtubeId'].unique().tolist()

                if len(test_data_video_id) > 0:

                    test_highlight_features_paths = []

                    for v_id in test_data_video_id:
                        test_highlight_features_paths += glob.glob(
                            '{}/{}_{}/highlight/*/*.npy'.format(self.video_feature_dir, v_id, user_id))

                    test_non_highlight_features_paths = []
                    for v_id in test_data_video_id:
                        test_non_highlight_features_paths += glob.glob(
                            '{}/{}_{}/non_highlight/*/*.npy'.format(self.video_feature_dir, v_id, user_id))

                    user_history_feature_path = os.path.join(self.user_history_feature_dir, '{}.npy'.format(user_id))
                    word_feature_representation_path = os.path.join(self.glove_feature_dir,
                                                                    '{}.p'.format(test_data_video_id[0]))

                    for positive in test_highlight_features_paths:
                        for negative in test_non_highlight_features_paths:
                            str_row = "{},{},{},{} {}".format(positive.lstrip(self.base_data_dir),
                                                              negative.lstrip(self.base_data_dir),
                                                              user_history_feature_path.lstrip(self.base_data_dir),
                                                              word_feature_representation_path,
                                                              os.linesep)
                            csvfile.write(str_row)


if __name__ == '__main__':
    obj = CreateTrainingDatasetFile()
    # obj.execute_train()
    obj.execute_test()
