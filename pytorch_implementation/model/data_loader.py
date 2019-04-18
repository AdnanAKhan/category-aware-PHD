import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from pprint import pprint


class HighlightDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Store the filenames of the extracted features to use. Specifies transforms to apply on the data.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on the data. basically just converts the
            data into tensor.
        """

        self.base_data_dir = data_dir
        self.mode = mode

        if mode == 'train':
            self.dataset_df = pd.read_csv(os.path.join(self.base_data_dir, 'train_dataset_pytorch.csv'), header=0)
            self.dataset_df = self.dataset_df[:int(len(self.dataset_df) * .8)]
        elif mode == 'test':
            self.dataset_df = pd.read_csv(os.path.join(self.base_data_dir, 'test_dataset_pytorch.csv'), header=0)
        elif mode == 'val':
            self.dataset_df = pd.read_csv(os.path.join(self.base_data_dir, 'train_dataset_pytorch.csv'), header=0)
            self.dataset_df = self.dataset_df[int(len(self.dataset_df) * .8):]
        else:
            print('select proper dataset')
            exit()
        self.highlight_segments = []
        self.non_highlight_segments = []
        self.textual_features = []
        self.user_history_features = []

        self.labels = []  # no labels available for highlight dataset.

        for ind, row in self.dataset_df.iterrows():
            self.highlight_segments.append(row['highlight'])
            self.non_highlight_segments.append(row['non_highlight'])
            self.textual_features.append(row['word_vector'].strip())
            self.user_history_features.append(row['user_history'])


        self.transform = True

    def __len__(self):
        # return size of dataset
        return len(self.highlight_segments)

    def __getitem__(self, idx):
        """
        Fetch index idx row and labels from dataset. Perform transforms on the data.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            segment feature: (Tensor) transformed numpy array (4096,0)
            non segment feature: (Tensor) transformed numpy array (4096,0)
            textual feature: (Tensor) transformed numpy array (100,0)
            user_history_features: (Tensor) (4096,0)
            label: returns 0.
        """
        highlight_npy = np.loadtxt(os.path.join(self.base_data_dir, self.highlight_segments[idx]), dtype=float)
        non_highlight_npy = np.loadtxt(os.path.join(self.base_data_dir, self.non_highlight_segments[idx]), dtype=float)
        text_feature_npy = np.loadtxt(os.path.join(self.base_data_dir, self.textual_features[idx]), dtype=float)
        user_history = np.loadtxt(os.path.join(self.base_data_dir, self.user_history_features[idx]), dtype=float)

        if self.transform:
            highlight_npy = torch.from_numpy(highlight_npy.reshape(4096, 1))
            non_highlight_npy = torch.from_numpy(non_highlight_npy.reshape(4096, 1))
            text_feature_npy = torch.from_numpy(text_feature_npy.reshape(100, 1))
            user_history = torch.from_numpy(user_history.reshape(4096, 1))

        return highlight_npy, non_highlight_npy, text_feature_npy, user_history


class DistanceNetDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Store the filenames of the extracted features to use. Specifies transforms to apply on the data.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on the data. basically just converts the
            data into tensor.
        """

        self.base_data_dir = data_dir
        self.mode = mode

        if mode == 'train':
            self.dataset_df = pd.read_csv(os.path.join(self.base_data_dir, 'train_dataset_pytorch.csv'), header=0)
            self.dataset_df = self.dataset_df[:int(len(self.dataset_df) * .8)]
        elif mode == 'test':
            self.dataset_df = pd.read_csv(os.path.join(self.base_data_dir, 'test_dataset_pytorch.csv'), header=0)
        elif mode == 'val':
            self.dataset_df = pd.read_csv(os.path.join(self.base_data_dir, 'train_dataset_pytorch.csv'), header=0)
            self.dataset_df = self.dataset_df[int(len(self.dataset_df) * .8):]
        else:
            print('select proper dataset')
            exit()
        self.highlight_segments = []
        self.non_highlight_segments = []
        self.user_history_features = []
        self.textual_features = []

        self.labels = []  # no labels available for highlight dataset.

        for ind, row in self.dataset_df.iterrows():
            self.highlight_segments.append(row['highlight'])
            self.non_highlight_segments.append(row['non_highlight'])
            self.user_history_features.append(row['user_history'])
            self.textual_features.append(row['word_vector'].strip())

        self.transform = True

    def __len__(self):
        return len(self.highlight_segments)

    def __getitem__(self, idx):
        """
        Fetch index idx row and labels from dataset. Perform transforms on the data.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            segment feature: (Tensor) transformed numpy array (4096,0)
            non segment feature: (Tensor) transformed numpy array (4096,0)
            textual feature: (Tensor) transformed numpy array (100,0)
            user_history_features: (Tensor) (4096,0)
            label: returns 0.
        """
        highlight_npy = np.loadtxt(os.path.join(self.base_data_dir, self.highlight_segments[idx]), dtype=float)
        non_highlight_npy = np.loadtxt(os.path.join(self.base_data_dir, self.non_highlight_segments[idx]), dtype=float)
        user_history = np.loadtxt(os.path.join(self.base_data_dir, self.user_history_features[idx]), dtype=float)
        text_feature_npy = np.loadtxt(os.path.join(self.base_data_dir, self.textual_features[idx]), dtype=float)

        if self.transform:
            highlight_npy = torch.from_numpy(highlight_npy.reshape(4096, 1))
            non_highlight_npy = torch.from_numpy(non_highlight_npy.reshape(4096, 1))
            user_history = torch.from_numpy(user_history.reshape(4096, 1))
            text_feature_npy = torch.from_numpy(text_feature_npy.reshape(100, 1))

        highlight_distance = torch.nn.functional.cosine_similarity(highlight_npy, user_history, dim=1, eps=1e-8)
        non_highlight_distance = torch.nn.functional.cosine_similarity(non_highlight_npy, user_history, dim=1, eps=1e-8)

        return highlight_npy, highlight_distance, non_highlight_npy, non_highlight_distance, text_feature_npy


def fetch_data_loader(types, data_dir, params, dataset_model='HighlightDataset'):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyper parameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            if split == 'train':
                if dataset_model == 'HighlightDataset':
                    dl = DataLoader(HighlightDataset(data_dir, mode='train'),
                                    batch_size=params.batch_size,
                                    shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
                elif dataset_model == 'DistanceNetDataset':
                    dl = DataLoader(DistanceNetDataset(data_dir, mode='train'),
                                    batch_size=params.batch_size,
                                    shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
                else:
                    pass

            elif split == 'val':
                if dataset_model == 'HighlightDataset':
                    dl = DataLoader(HighlightDataset(data_dir, mode='val'),
                                    batch_size=params.batch_size,
                                    shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
                elif dataset_model == 'DistanceNetDataset':
                    dl = DataLoader(DistanceNetDataset(data_dir, mode='val'),
                                    batch_size=params.batch_size,
                                    shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
                else:
                    pass

            elif split == 'test':
                if dataset_model == 'HighlightDataset':
                    dl = DataLoader(HighlightDataset(data_dir, mode='test'),
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
                elif dataset_model == 'DistanceNetDataset':
                    dl = DataLoader(DistanceNetDataset(data_dir, mode='test'),
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
                else:
                    pass

            else:
                # not handling the test in this case.
                dl = None

            if dl:
                dataloaders[split] = dl

    return dataloaders


def test_higlight_dataset_loader():
    """
    tests the shape and type of the dataset
    :return:
    """
    transformed_dataset = HighlightDataset(data_dir='/home/adnankhan/PycharmProjects/HighlightDetection/Data/',
                                           transform=None)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        highlight_npy, non_highlight_npy, text_feature, user_history = sample
        print(
            """
            highlight feature shape {} and type {}
            non highlight feature shape {} and type {}
            text feature shape {} and type {}
            user history feature shape {} and type {}
            """.format(highlight_npy.shape, type(highlight_npy),
                       non_highlight_npy.shape, type(non_highlight_npy),
                       text_feature.shape, type(text_feature),
                       user_history.shape, type(user_history)
                       )
        )

        if i == 1:
            break


def test_distance_dataset_loader():
    """
    tests the shape and type of the dataset
    :return:
    """
    transformed_dataset = DistanceNetDataset(data_dir='/home/adnankhan/PycharmProjects/HighlightDetection/Data/',
                                             transform=None)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        highlight_npy, highlight_distance, non_highlight_npy, non_highlight_distance, test_feature_npy = sample
        print(
            """
            highlight feature shape {} and type {}
            highlight  distance feature shape {} and type {}
            non highlight feature shape {} and type {}
            non highlight distance feature shape {} and type {}
            text feature shape {} and type {}
            """.format(highlight_npy.shape, type(highlight_npy),
                       highlight_distance.shape, type(highlight_distance),
                       non_highlight_npy.shape, type(non_highlight_npy),
                       non_highlight_distance.shape, type(non_highlight_distance),
                       test_feature_npy.shape, type(test_feature_npy)
                       )
        )

        if i == 1:
            break


if __name__ == '__main__':
    # obj = DistanceNetDataset(data_dir='/home/adnankhan/PycharmProjects/HighlightDetection/Data/')
    test_distance_dataset_loader()
