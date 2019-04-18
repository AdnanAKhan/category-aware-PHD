"""Evaluates the model on the test data"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from utils import utils as utils
from model.net import PhdGifNet, loss_fn, metrics, accuracy
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/home/adnankhan/PycharmProjects/HighlightDetection/',
                    help="Directory containing the implementations")
parser.add_argument('--data_dir', default='/home/adnankhan/PycharmProjects/HighlightDetection/Data/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir',
                    default='/home/adnankhan/PycharmProjects/HighlightDetection/pytorch_implementation/experiments/test_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def evaluate(model,
             loss_fn,
             dataloader,
             metrics,
             params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyper parameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset

    for i, (highlight_batch, highlight_distance_batch, non_highlight_batch, non_highlight_distance_batch, text_feature_batch) in enumerate(dataloader):
        highlight_batch = highlight_batch.reshape(highlight_batch.shape[0], -1).float()
        highlight_distance_batch = highlight_batch.reshape(highlight_distance_batch.shape[0], -1).float()
        non_highlight_batch = non_highlight_batch.reshape(non_highlight_batch.shape[0], -1).float()
        non_highlight_distance_batch = non_highlight_batch.reshape(non_highlight_distance_batch.shape[0], -1).float()
        text_feature_batch = text_feature_batch.reshape(text_feature_batch.shape[0], -1).float()

        positive_batch = torch.cat((highlight_batch, highlight_distance_batch, text_feature_batch), dim=1)
        negative_batch = torch.cat((non_highlight_batch, non_highlight_distance_batch, text_feature_batch), dim=1)

        # move to GPU if available
        if params.cuda:
            positive_batch, negative_batch = positive_batch.cuda(async=True), negative_batch.cuda(async=True)
            device = torch.device("cuda")

        positive_batch, negative_batch = Variable(positive_batch), Variable(negative_batch)

        positive_batch_output = model(positive_batch)
        negative_batch_output = model(negative_batch)

        if params.cuda:
            loss = loss_fn(positive_batch_output, negative_batch_output,
                           torch.ones(positive_batch.shape[0], 1, device=device))
        else:
            loss = loss_fn(positive_batch_output, negative_batch_output,
                           torch.ones(positive_batch.shape[0], 1))

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](positive_batch_output, negative_batch_output) for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    """
    I need to understand this block more. 
    - Just computing the mean for each matrices in the metrics dictionary from net.py file
    """
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_data_loader(['val'], args.data_dir, params)
    test_dl = dataloaders['val']

    logging.info("getting the test dataloader - done.")

    # Define the model
    model = PhdGifNet().cuda() if params.cuda else PhdGifNet()

    loss_fn = loss_fn
    metrics = metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
