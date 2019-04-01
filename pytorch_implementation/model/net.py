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

    def num_flat_features(self, x):
        x = x.view(-1, self.num_flat_features(x))
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    loss = nn.MarginRankingLoss(reduction='mean')  # possible options are 'sum'
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
