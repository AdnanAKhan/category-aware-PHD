"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import model.data_loader as data_loader
from model.net import PhdGifNet, accuracy, loss_fn, metrics
from utils import utils as utils
from evaluate_phd_gif import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/home/adnankhan/PycharmProjects/HighlightDetection/',
                    help="Directory containing the implementations")
parser.add_argument('--data_dir', default='/home/adnankhan/PycharmProjects/HighlightDetection/Data/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir',
                    default='/home/adnankhan/PycharmProjects/HighlightDetection/pytorch_implementation/experiments/phd_gif_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summary = []
    loss_over_batch = []
    loss_avg = utils.RunningAverage()

    for i, (highlight_batch, non_highlight_batch, text_feature_batch, user_history_batch) in enumerate(dataloader):
        highlight_batch = highlight_batch.reshape(highlight_batch.shape[0], -1).float()
        non_highlight_batch = non_highlight_batch.reshape(non_highlight_batch.shape[0], -1).float()
        user_history_batch = user_history_batch.reshape(user_history_batch.shape[0], -1).float()

        positive_batch = torch.cat((highlight_batch, user_history_batch), dim=1)
        negative_batch = torch.cat((non_highlight_batch, user_history_batch), dim=1)

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

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](positive_batch_output, negative_batch_output)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            # logging.info("- Batch loss: {}".format(summary_batch['loss']))
            summary.append(summary_batch)

        loss_over_batch.append(loss.item())
        # update the average loss
        loss_avg.update(loss.item())

    metrics_mean = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return np.array(loss_over_batch)


def train_and_evaluate(model,
                       train_dataloader,
                       val_dataloader,
                       optimizer,
                       loss_fn,
                       metrics,
                       params,
                       model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    loss_over_epoch=[]

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        loss_over_batch = train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        loss_over_epoch.append(loss_over_batch)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        #
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    return loss_over_epoch


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    print("cuda available : {}".format(params.cuda))

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_data_loader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("dataset loading - done.")

    # Define the model and optimizer
    model = PhdGifNet().cuda() if params.cuda else PhdGifNet()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = loss_fn
    metrics = metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    loss_over_epoch = train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)

    logging.info("saving the loss in {}".format(args.model_dir))
    np.savetxt(os.path.join(args.model_dir, 'train_loss.npy'), np.array(loss_over_epoch))