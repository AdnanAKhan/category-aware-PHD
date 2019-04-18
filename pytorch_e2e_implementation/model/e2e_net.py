import torch.nn as nn
import torch.nn.functional as F
import os


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
    loss = nn.MarginRankingLoss(
        reduction='mean', margin=2)  # possible options are 'sum'
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
    # get difference between the highlight segments and non highlight segments
    diff = positive_outputs - negative_output
    predictions = diff[diff > margin]
    return predictions.shape[0] / sample_length


metrics = {
    'accuracy': accuracy,
}


class CustomC3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(CustomC3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(
            64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(
            128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(
            256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.custom_dropout = nn.Dropout(p=0.5)
        self.custom_fc1 = nn.Linear(8192, 512)
        self.custom_fc2 = nn.Linear(512, 128)
        self.custom_score = nn.Linear(128, 1)

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

        h = self.relu(self.custom_fc1(h))
        h = self.custom_dropout(h)
        h = self.relu(self.custom_fc2(h))
        h = self.custom_dropout(h)
        score = self.score(h)

        return score
