import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, brier_score_loss


import torch
from torch.utils.data import Dataset
from torchmetrics.classification import (BinaryROC,
                                         BinaryRecall,
                                         BinaryF1Score,
                                         BinaryAUROC,
                                         BinaryPrecisionRecallCurve)




# for reproducible results
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.determenistic=True

random.seed(random_seed)
def seed_worker(worker_id):
    worker_seed = random_seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(random_seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #specifically required for reproducibility with CuBLABS and CUDA
os.environ["PYTHONHASHSEED"] = str(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(np.ceil(len(y) / batch_size))
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = 42
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches


# Define custom DataLoaders
# train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# test data
class TestData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f'Train time on {device}: {total_time:.3f} seconds')
    return total_time



def auc_fn(y_true, y_probs):
    """
    Get FPR, TPR and thresholds for the AUC curve
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    metric = BinaryROC(thresholds=None).to(device)
    fpr, tpr, thresholds = metric(y_probs, y_true.type(torch.int))
    J = tpr.detach().cpu().numpy() - fpr.detach().cpu().numpy()
    ix = np.argmax(J)
    best_thresh = thresholds[ix].item()
    return best_thresh


def auroc_fn(y_true, y_probs):
    """
    Calculate AUROC
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    metric = BinaryAUROC(thresholds=None).to(device)
    return (metric(y_probs, y_true) * 100).item()


def auc_pr_fn(y_true, y_probs):
    """
    Calculate AUC of the PR curve
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    metric = BinaryPrecisionRecallCurve(thresholds=None).to(device)
    precision, recall, thresholds = metric(y_probs, y_true.type(torch.int))
    precision, recall = precision.detach().cpu().numpy(), recall.detach().cpu().numpy()
    return auc(recall, precision) * 100


def recall_fn(y_true, y_pred, y_probs):
    """
    Calculate recall
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_pred: A tensor of predicted values, in credit scoring 0 or 1 values indicating if the borrower would be liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    best_thresh = auc_fn(y_true, y_probs)
    metric = BinaryRecall(threshold=best_thresh).to(device)
    return (metric(y_pred, y_true) * 100).item()


def brier_fn(y_true, y_probs):
    """
    Calculate Brier Score
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    y_true, y_probs = y_true.detach().cpu().numpy(), y_probs.detach().cpu().numpy()
    return brier_score_loss(y_true, y_probs) * 100


def ks_fn(y_true, y_probs):
    """
    Calculate the KS-Statistic
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    y_true = pd.Series(y_true.detach().cpu().numpy())
    y_probs = pd.Series(y_probs.detach().cpu().numpy())
    y_true = pd.concat([y_true, y_probs], axis=1)
    y_true.columns = ['y_test_class_actual', 'y_hat_test_proba']
    y_true.sort_values('y_hat_test_proba', inplace=True)
    y_true.reset_index(drop=True, inplace=True)
    y_true['Cumulative N Population'] = y_true.index + 1
    y_true['Cumulative N Bad'] = y_true['y_test_class_actual'].cumsum()
    y_true['Cumulative N Good'] = y_true['Cumulative N Population'] - y_true['Cumulative N Bad']
    y_true['Cumulative Perc Population'] = y_true['Cumulative N Population'] / y_true.shape[0]
    y_true['Cumulative Perc Bad'] = y_true['Cumulative N Bad'] / y_true['y_test_class_actual'].sum()
    y_true['Cumulative Perc Good'] = y_true['Cumulative N Good'] / (
                y_true.shape[0] - y_true['y_test_class_actual'].sum())
    KS = max(y_true['Cumulative Perc Good'] - y_true['Cumulative Perc Bad'])

    return KS * 100


def prob_diff_fn(y_true, y_probs):
    """
    Calculate the difference between the median predicted probabilities of the two classes
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    y_true, y_probs = pd.Series(y_true.detach().cpu().numpy()), pd.Series(y_probs.detach().cpu().numpy())
    y_true = pd.concat([y_true, y_probs], axis=1)
    y_true.columns = ['y_test_class_actual', 'y_hat_test_proba']
    prob_density_diff = y_true.loc[y_true['y_test_class_actual'] == 1, 'y_hat_test_proba'].median() - y_true.loc[y_true['y_test_class_actual'] == 0, 'y_hat_test_proba'].median()
    return prob_density_diff * 100


def f1_score_fn(y_true, y_pred, y_probs):
    """
    Calculate F1-Score
    :param y_true: A tensor of ground truth values, in credit scoring 0 or 1 values indicating if the borrower was liquidated in some period
    :param y_pred: A tensor of predicted values, in credit scoring 0 or 1 values indicating if the borrower would be liquidated in some period
    :param y_probs: A tensor of predicted probabilities, in credit scoring the probability of the borrower being liquidated in some period (between 0 and 1)
    :return:
    """
    best_thresh = auc_fn(y_true, y_probs)
    metric = BinaryF1Score(threshold=best_thresh).to(device)
    return (metric(y_pred, y_true) * 100).item()


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               epochs: int,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               auc_fn,
               recall_fn,
               f1_score_fn,
               auroc_fn,
               brier_fn,
               auc_pr_fn,
               ks_fn,
               prob_diff_fn,
               device: torch.device = device,
               simple_submission = False):
    """
    Training function for the model provided in the starter kit
    :param model:
    :param data_loader:
    :param epochs:
    :param loss_fn:
    :param optimizer:
    :param auc_fn:
    :param recall_fn:
    :param f1_score_fn:
    :param auroc_fn:
    :param brier_fn:
    :param auc_pr_fn:
    :param ks_fn:
    :param prob_diff_fn:
    :param device:
    :return:
    """
    warnings.simplefilter('ignore')
    train_loss, train_rec, train_f1, train_auroc, train_brier, train_aucpr, train_ks, train_prob_diff = 0, 0, 0, 0, 0, 0, 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to the available device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_probs = model(X).squeeze()  # probabilities
        y_pred = torch.round(y_probs)  # predicted labels

        # Calculate loss & other metrics
        loss = loss_fn(y_probs, y)
        train_loss += loss
        train_rec += recall_fn(y_true=y, y_pred=y_pred, y_probs=y_probs)
        train_f1 += f1_score_fn(y_true=y, y_pred=y_pred, y_probs=y_probs)
        train_auroc += auroc_fn(y_true=y, y_probs=y_probs)
        train_brier += brier_fn(y_true=y, y_probs=y_probs)
        train_aucpr += auc_pr_fn(y_true=y, y_probs=y_probs)
        train_ks += ks_fn(y_true=y, y_probs=y_probs)
        train_prob_diff += prob_diff_fn(y_true=y, y_probs=y_probs)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Calculate loss and other metrics per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_rec /= len(data_loader)
    train_f1 /= len(data_loader)
    train_auroc /= len(data_loader)
    train_brier /= len(data_loader)
    train_aucpr /= len(data_loader)
    train_ks /= len(data_loader)
    train_prob_diff /= len(data_loader)

    if epochs % 5 == 0 and not simple_submission:
        print(
            f'Training metrics:\nLoss: {train_loss:.5f} | Recall: {train_rec:.2f}% | F1-Score: {train_f1:.2f}% | AUROC: {train_auroc:.2f}% | Brier Score: {train_brier:.2f}% | AUC PR: {train_aucpr:.2f}% | KS-Statistic: {train_ks:.2f}% | Pred Prob Diff: {train_prob_diff:.2f}%')
    return {'loss': train_loss.item(),
            'recall': train_rec,
            'f1': train_f1,
            'auroc': train_auroc,
            'brier': train_brier,
            'aucpr': train_aucpr,
            'ks': train_ks,
            'prob_diff': train_prob_diff}


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              epochs: int,
              auc_fn,
              recall_fn,
              f1_score_fn,
              auroc_fn,
              brier_fn,
              auc_pr_fn,
              ks_fn,
              prob_diff_fn,
              device: torch.device = device,
              simple_submission = False):
    """
    Test step for the example model in the starter kit
    :param data_loader:
    :param model:
    :param loss_fn:
    :param epochs:
    :param auc_fn:
    :param recall_fn:
    :param f1_score_fn:
    :param auroc_fn:
    :param brier_fn:
    :param auc_pr_fn:
    :param ks_fn:
    :param prob_diff_fn:
    :param device:
    :return:
    """
    warnings.simplefilter('ignore')
    test_loss, test_rec, test_f1, test_auroc, test_brier, test_aucpr, test_ks, test_prob_diff = 0, 0, 0, 0, 0, 0, 0, 0
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the applicable device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_probs = model(X).squeeze()  # probabilities
            y_pred = torch.round(y_probs)  # predicted labels

            # Calculate loss and other metrics
            test_loss += loss_fn(y_probs, y)
            test_rec += recall_fn(y_true=y, y_pred=y_pred, y_probs=y_probs)
            test_f1 += f1_score_fn(y_true=y, y_pred=y_pred, y_probs=y_probs)
            test_auroc += auroc_fn(y_true=y, y_probs=y_probs)
            test_brier += brier_fn(y_true=y, y_probs=y_probs)
            test_aucpr += auc_pr_fn(y_true=y, y_probs=y_probs)
            test_ks += ks_fn(y_true=y, y_probs=y_probs)
            test_prob_diff += prob_diff_fn(y_true=y, y_probs=y_probs)

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_rec /= len(data_loader)
        test_f1 /= len(data_loader)
        test_auroc /= len(data_loader)
        test_brier /= len(data_loader)
        test_aucpr /= len(data_loader)
        test_ks /= len(data_loader)
        test_prob_diff /= len(data_loader)

        if epochs % 5 == 0 and not simple_submission:
            print(
                f'Testing metrics:\nLoss: {test_loss:.5f} | Recall: {test_rec:.2f}% | F1-Score: {test_f1:.2f}% | AUROC: {test_auroc:.2f}% | Brier Score: {test_brier:.2f}% | AUC PR: {test_aucpr:.2f}% | KS-Statistic: {test_ks:.2f}% | Pred Prob Diff: {test_prob_diff:.2f}%')
        return {'loss': test_loss.item(),
                'recall': test_rec,
                'f1': test_f1,
                'auroc': test_auroc,
                'brier': test_brier,
                'aucpr': test_aucpr,
                'ks': test_ks,
                'prob_diff': test_prob_diff}


# model evaluation/validation function
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               auc_fn,
               recall_fn,
               f1_score_fn,
               auroc_fn,
               brier_fn,
               auc_pr_fn,
               ks_fn,
               prob_diff_fn,
               device: torch.device = device):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        other model validation metrics
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """

    warnings.simplefilter('ignore')
    loss, rec, f1, auroc, brier, aucpr, ks, prob_diff = 0, 0, 0, 0, 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Make predictions with the model
            y_probs = model(X).squeeze()  # probabilities
            y_pred = torch.round(y_probs)  # predicted labels

            # Accumulate the loss and other metrics values per batch
            loss += loss_fn(y_probs, y)
            rec += recall_fn(y_true=y, y_pred=y_pred, y_probs=y_probs)
            f1 += f1_score_fn(y_true=y, y_pred=y_pred, y_probs=y_probs)
            auroc += auroc_fn(y_true=y, y_probs=y_probs)
            brier += brier_fn(y_true=y, y_probs=y_probs)
            aucpr += auc_pr_fn(y_true=y, y_probs=y_probs)
            ks += ks_fn(y_true=y, y_probs=y_probs)
            prob_diff += prob_diff_fn(y_true=y, y_probs=y_probs)

        # Scale loss and other metrics to find the averages per batch
        loss /= len(data_loader)
        rec /= len(data_loader)
        f1 /= len(data_loader)
        auroc /= len(data_loader)
        brier /= len(data_loader)
        aucpr /= len(data_loader)
        ks /= len(data_loader)
        prob_diff /= len(data_loader)

    return {'model_name': model.__class__.__name__,
            # only works when model was created with a class and is not compiled
            'model_loss': loss.item(),
            'model_rec': rec,
            'model_f1': f1,
            'model_auroc': auroc,
            'model_brier': brier,
            'model_aucpr': aucpr,
            'model_ks': ks,
            'model_prob_diff': prob_diff}


# Plot the loss curves
def plot_loss_curves(epoch_count, train_loss_values, test_loss_values):
    """
    Plot the loss curves for the training and test set loss, where x is the epoch and y is the loss
    :param epoch_count:
    :param train_loss_values:
    :param test_loss_values:
    :return:
    """
    plt.plot(epoch_count, train_loss_values, label='Train loss')
    plt.plot(epoch_count, test_loss_values, label='Test loss')
    plt.title('Training and test loss curves')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend();


class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        # number of epochs to allow for no improvement before stopping the execution
        self.patience = patience
        # the minimum change to be counted as improvement
        self.min_delta = min_delta
        # count the number of times the validation accuracy not improving
        self.counter = 0
        self.min_validation_loss = np.inf

    # return True when encountering _patience_ times decrease in validation loss
    def early_stop_check(self, validation_loss):
        if ((validation_loss + self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            # reset the counter if validation loss decreased at least by min_delta
            self.counter = 0
        elif ((validation_loss + self.min_delta) > self.min_validation_loss):
            # increase the counter if validation loss is not decreased by the min_delta
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False