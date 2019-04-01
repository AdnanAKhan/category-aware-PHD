""" How to use C3D network. """
import numpy as np
import torch
from torch.autograd import Variable
import os
import random
from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

import torch.nn as nn

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs


class ExtractC3D:
    """
    Extracts C3D feature from images
    """

    def __init__(self):
        self.DATA_DIR = '/home/adnankhan/dataset/'
        self.SEGMENT_DIR = os.path.join(self.DATA_DIR, 'video_segments')
        self.FEATURE_DIR = os.path.join(self.DATA_DIR, 'feature_c3d', 'video_features')
        if not os.path.exists(self.FEATURE_DIR):
            os.makedirs(self.FEATURE_DIR)
        self.SEQUENCE_LENGTH = 16
        self.NUMBER_OF_SAMPLES = 10
        random.seed(2424)

        self.params = {}

        # use GPU if available
        self.params['cuda'] = torch.cuda.is_available()  # use GPU is available

        # Set the random seed for reproducible experiments
        torch.manual_seed(230)
        if self.params['cuda']:
            torch.cuda.manual_seed(230)

        # get network pre-trained model
        self.net = C3D()
        self.net.load_state_dict(torch.load('c3d.pickle'))
        if self.params['cuda']:
            self.net.cuda()

        self.net.eval()

    def iterator_over_sampled_frames(self):
        base_folders = os.listdir(self.SEGMENT_DIR)

        for folder in base_folders[:1]:
            # segment folders
            if os.path.exists(os.path.join(self.SEGMENT_DIR, folder, 'highlight')):
                for segment_folder in os.listdir(os.path.join(self.SEGMENT_DIR, folder, 'highlight')):
                    segment_path = os.path.join(self.SEGMENT_DIR, folder, 'highlight', segment_folder)
                    dest_path = os.path.join(self.FEATURE_DIR, folder, 'highlight', segment_folder)
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)
                        self.extract_feature_from_video(segment_path, dest_path)

            # Non segment folders
            if os.path.exists(os.path.join(self.SEGMENT_DIR, folder, 'non_highlight')):
                for segment_folder in os.listdir(os.path.join(self.SEGMENT_DIR, folder, 'non_highlight')):
                    segment_path = os.path.join(self.SEGMENT_DIR, folder, 'non_highlight', segment_folder)
                    dest_path = os.path.join(self.FEATURE_DIR, folder, 'non_highlight', segment_folder)
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)
                        self.extract_feature_from_video(segment_path, dest_path)

    def extract_feature_from_video(self, src_path, dest_path):
        clip = sorted(glob(join(src_path, '*.jpg')))
        if len(clip) > 0:
            clip_segments = []
            number_of_frames = len(clip)

            for i in range(0, number_of_frames, self.SEQUENCE_LENGTH):
                if i + self.SEQUENCE_LENGTH > number_of_frames:
                    clip_segments.append(clip[number_of_frames - self.SEQUENCE_LENGTH: number_of_frames])
                else:
                    clip_segments.append(clip[i:i + self.SEQUENCE_LENGTH])

            random.shuffle(clip_segments)

            for ind, segment in enumerate(clip_segments[0:self.NUMBER_OF_SAMPLES]):
                numpy_data = ExtractC3D.get_segment_numpy_array(segment)
                features = self.extract(numpy_data)
                np.savetxt(os.path.join(dest_path, '{}.npy'.format(ind)), features)

    @staticmethod
    def get_segment_numpy_array(segment, verbose=False):

        """
           Loads a clip to be fed to C3D for classification.
           Parameters
           ----------
           segment: str
               the name of the clip (subfolder in 'data').
           verbose: bool
               if True, shows the unrolled clip (default is True).

           Returns
           -------
           Tensor
               a pytorch batch (n, ch, fr, h, w).
           """

        clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in segment])
        clip = clip[:, :, 44:44 + 112, :]  # crop centrally

        if verbose:
            clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
            io.imshow(clip_img.astype(np.uint8))
            io.show()

        clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
        clip = np.expand_dims(clip, axis=0)  # batch axis
        clip = np.float32(clip)

        return torch.from_numpy(clip)

    def extract(self, data):
        feature_fc7 = torch.zeros((1, 4096))

        def copy_data(m, i, o):
            feature_fc7.copy_(o.data)

        h = self.net.fc7.register_forward_hook(copy_data)

        X = Variable(data)
        if self.params['cuda']:
            X = X.cuda()
        prediction = self.net(X)
        # prediction = prediction.data.cpu().numpy()
        return feature_fc7.data.cpu().numpy()

    def test(self):
        path = os.path.join(self.FEATURE_DIR, 'kDqG9h-C6p0_13584', 'highlight', '0')
        folders = os.listdir(path)
        npy_array = np.loadtxt(os.path.join(path, folders[0]))
        print(npy_array.shape)

    def run(self):
        self.iterator_over_sampled_frames()


if __name__ == '__main__':
    obj = ExtractC3D()
    # obj.run()
    obj.test()
