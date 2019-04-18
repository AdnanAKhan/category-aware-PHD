import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple two layer neural network based on
    Link: https://github.com/gyglim/video2gif_code/blob/master/video2gif/model.py
    Their c3d portion is removed as we already calculated the c3d features on the segment
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8292, 512)
        self.fc2 = nn.Linear(512, 128)
        self.score = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.score(x)
        return x

class PhdGifNet(nn.Module):
    """
    Simple two layer neural network based on
    Link: https://github.com/gyglim/video2gif_code/blob/master/video2gif/model.py
    Their c3d portion is removed as we already calculated the c3d features on the segment
    """

    def __init__(self):
        super(PhdGifNet, self).__init__()
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 128)
        self.score = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.score(x)
        return x

class DistanceNet(nn.Module):
    """
    Simple two layer neural network based on
    Link: https://github.com/gyglim/video2gif_code/blob/master/video2gif/model.py
    Their c3d portion is removed as we already calculated the c3d features on the segment
    """

    def __init__(self):
        super(DistanceNet, self).__init__()
        self.fc1 = nn.Linear(8292, 512)
        self.fc2 = nn.Linear(512, 128)
        self.score = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.score(x)
        return x

def loss_fn(positive_outputs, negative_outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Loss documentation
    Link: https://pytorch.org/docs/stable/nn.html#torch.nn.MarginRankingLoss
    Args:
        outputs: (Variable) dimension batch_size x 2 - [output of the model x 2]
        labels: (Variable) dimension batch_size x 2, where each element is a value in [[1, 0], [1,0]]
    Returns:
        loss (Variable): Margin Ranking Loss for all images in the batch
    """
    loss = nn.MarginRankingLoss(reduction='mean', margin=2)  # possible options are 'sum'
    return loss(positive_outputs, negative_outputs, labels)


def accuracy(positive_outputs, negative_output, margin=0.0):
    """
    Compute the accuracy, given the outputs and labels for all data.
    Args:
        positive_outputs: (torch tensor) dimension batch_size x 2
        negative_output: (torch.tensor) dimension batch_size x 2
        margin: specifies how much difference we want in the ranking score.
    Returns: (float) accuracy

    defination: accuracy here:
        number of data where rank of highlight segment is higher than rank of non highlight segments
    """
    sample_length = positive_outputs.shape[0]
    diff = positive_outputs - negative_output  # get difference between the highlight segments and non highlight segments
    predictions = diff[diff > margin]
    return predictions.shape[0] / sample_length


metrics = {
    'accuracy': accuracy,
}


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