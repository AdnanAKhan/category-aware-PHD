import os
import pandas as pd
from pprint import pprint
import pickle
import numpy as np
import bcolz


class ExtractGloveFeature:
    def __init__(self):
        self.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'Data')
        self.DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     'Data', 'text_features')

        self.train_df = pd.read_csv(os.path.join(self.DATA_DIR, 'train_dataset.csv'))
        self.test_df = pd.read_csv(os.path.join(self.DATA_DIR, 'test_dataset.csv'))

        vectors = bcolz.open(f'6B.100.dat')[:]
        words = pickle.load(open(f'6B.100_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'6B.100_idx.pkl', 'rb'))

        self.glove = {w: vectors[word2idx[w]] for w in words}
        self.key_error_count = 0

    def create_representation(self):
        for grp, data in self.test_df.groupby(['youtubeId']):
            datum = data.iloc[0]
            dest_file_name = '{}.npy'.format(datum['youtubeId'])
            keyword = datum['keyword']
            vector_representation = self.get_glove_vector(keyword)

            np.savetxt(os.path.join(self.DEST_DIR, dest_file_name),vector_representation)

        for grp, data in self.train_df.groupby(['youtubeId']):
            datum = data.iloc[0]
            dest_file_name = '{}.npy'.format(datum['youtubeId'])
            keyword = datum['keyword']
            vector_representation = self.get_glove_vector(keyword)

            np.savetxt(os.path.join(self.DEST_DIR, dest_file_name), vector_representation)

    def get_glove_vector(self, inp):
        result = np.zeros((100,))

        for word in inp.split():
            try:
                result += self.glove[word]
            except KeyError:
                print('key error- {}'.format(word))
                self.key_error_count += 1

        return result

    @staticmethod
    def pre_process_glove_dict():
        """
        get pre processed vector.
        :return:
        """
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'6B.100.dat', mode='w')

        with open('glove.6B.100d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)

        vectors = bcolz.carray(vectors[1:].reshape((400001, 100)), rootdir=f'6B.100.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open('6B.100_words.pkl', 'wb'))
        pickle.dump(word2idx, open('6B.100_idx.pkl', 'wb'))


if __name__ == '__main__':
    obj = ExtractGloveFeature()
    # run the below line to initialize the glove vector.
    # ExtractGloveFeature.pre_process_glove_dict()
    obj.create_representation()
    print(obj.key_error_count)
