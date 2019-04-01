import pandas as pd
import os
import pickle


class VideoSegmentation:

    def __init__(self, train=True):
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        self.SAVE_DIR = os.path.join(self.DATA_DIR, 'segment_info')
        if train:
            self.df = pd.read_csv(os.path.join(self.DATA_DIR, 'train_dataset.csv'))
        else:
            self.df = pd.read_csv(os.path.join(self.DATA_DIR, 'test_dataset.csv'))

        self.DELTA = 0.03

    def process(self):
        for id, datum in self.df.groupby(['youtubeId', 'user_id']):
            video_id = datum.iloc[0]['youtubeId']
            user_id = datum.iloc[0]['user_id']
            total_duration = datum.iloc[0]['video_duration']

            highlight_durations = []
            non_highlight_durations = []

            for ind, val in datum.iterrows():
                duration_pair = {
                    'start': val['start'],
                    'end': val['start'] + val['duration']
                }
                highlight_durations.append(duration_pair)

            highlight_durations = sorted(highlight_durations, key=lambda k: k['start'], reverse=False)

            union_set = []

            for ind, data in enumerate(highlight_durations):
                if ind == 0:
                    union_set.append(data)
                else:
                    for i, n in enumerate(union_set):
                        if n['start'] < data['start'] and n['end'] > data['end']:
                            continue
                        elif data['start'] < n['end'] < data['end']:
                            n['end'] = data['end']
                            continue
                        elif n['start'] > data['start'] and n['end'] < data['end']:
                            n['end'] = data['end']
                            n['start'] = data['start']
                            continue
                        elif n['start'] > data['start'] and n['end'] > data['end']:
                            n['start'] = data['start']
                            continue
                        else:
                            if i == len(union_set) - 1:
                                union_set.append(data)
                                break

            highlight_durations = union_set

            non_high = {'start': 0.0, 'end': highlight_durations[0]['start'] - self.DELTA}
            non_highlight_durations.append(non_high)

            for i in range(len(highlight_durations)):
                if i + 1 == len(highlight_durations):
                    non_high = {'start': highlight_durations[i]['end'] + self.DELTA, 'end': total_duration}
                    non_highlight_durations.append(non_high)
                else:
                    non_high = {'start': highlight_durations[i]['end'] + self.DELTA,
                                'end': highlight_durations[i + 1]['start'] - self.DELTA}
                    non_highlight_durations.append(non_high)

            save_dict = {
                'highlight_segment': highlight_durations,
                'non_highlight_segment': non_highlight_durations,
                'total_duration': total_duration
            }

            pickle.dump(save_dict, open(os.path.join(self.SAVE_DIR, '{}_{}.p'.format(video_id, user_id)), 'wb'))


if __name__ == '__main__':
    obj = VideoSegmentation(train=False)
    obj.process()
