import os
import json
import csv
import pandas as pd
import random


class DatasetSelection:

    def __init__(self, save=True):
        """
        initialization of data directories
        """
        random.seed(3022)

        self.SAVE = save
        self.SPLIT = 0.80
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        self.DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data',
                                    'personalized-highlights-dataset')
        self.categories = {}
        self.keywords_to_video = {}
        self.video_to_keywords = {}

        self.selected_category_to_videos = {}

        with open(os.path.join(self.DATA_DIR, 'category.json'), 'r') as f:
            self.categories = json.load(f)

        with open(os.path.join(self.DATA_DIR, 'keywords_to_video_mapping.json'), 'r') as f:
            self.keywords_to_video = json.load(f)

        for k, v in self.categories.items():
            self.selected_category_to_videos[k] = self.keywords_to_video[k]

    def map_video_to_category(self):
        """
        creates video to keywords mapping
        :return:
        """
        for k, v in self.selected_category_to_videos.items():

            for v_id in v:
                if v_id in self.video_to_keywords:
                    if k not in self.video_to_keywords[v_id]:
                        self.video_to_keywords[v_id].append(k)
                else:
                    self.video_to_keywords[v_id] = [k]

        if self.SAVE:
            with open(os.path.join(self.DATA_DIR, 'video_id_to_keywords.json'), 'w') as f:
                json.dump(self.video_to_keywords, f)

    def create_dataset(self):
        rows_to_write = []

        with open(os.path.join(self.DATASET, 'testing.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                filename = row[0] + '.json'
                if filename in self.video_to_keywords:
                    keywords = self.video_to_keywords[filename]
                    new_row = row + keywords[:1]  # just picked the first category for simplicity
                    rows_to_write.append(new_row)
        with open(os.path.join(self.DATA_DIR, 'output_dataset.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            header_row = ['youtubeId', 'start', 'duration', 'user_id', 'video_duration', 'is_last', 'keyword']
            writer.writerow(header_row)
            for row in rows_to_write:
                writer.writerow(row)

    def prepare_dataset(self):
        df = pd.read_csv(os.path.join(self.DATA_DIR, 'output_dataset.csv'))
        processed_df = pd.DataFrame(columns=df.columns)
        unique_userids = df['user_id'].unique()

        for user_id in unique_userids:
            user_df = df[df['user_id'] == user_id]
            if user_df[user_df['is_last'] == True]['youtubeId'].count() == 0:
                youtube_list = user_df['youtubeId'].unique()
                random.shuffle(youtube_list)
                youtube_id_to_make_true = youtube_list[0]
                pd.set_option('mode.chained_assignment', None)
                user_df.loc[user_df['youtubeId'] == youtube_id_to_make_true, 'is_last'] = True
                processed_df = pd.concat([processed_df, user_df], ignore_index=True)
            else:
                processed_df = pd.concat([processed_df, user_df], ignore_index=True)

        unique_users = processed_df['user_id'].unique()
        unique_users = list(unique_users)
        random.shuffle(unique_users)
        split = int(len(unique_users) * self.SPLIT)
        train_users = unique_users[:split]
        train_df = processed_df[processed_df['user_id'].isin(train_users)]
        test_users = unique_users[split:]
        test_df = processed_df[processed_df['user_id'].isin(test_users)]

        # test case
        failed = False
        for grp, data in train_df.groupby(['user_id']):
            if data[data['is_last'] == True]['user_id'].count() == 0:
                failed=True

        if failed:
            print("Failed to have one is_last==true")
        else:
            train_df.to_csv(os.path.join(self.DATA_DIR, 'train_dataset.csv'), sep=',', index=False)
            test_df.to_csv(os.path.join(self.DATA_DIR, 'test_dataset.csv'), sep=',', index=False)


if __name__ == '__main__':
    obj = DatasetSelection(save=True)
    obj.map_video_to_category()
    obj.create_dataset()
    obj.prepare_dataset()
