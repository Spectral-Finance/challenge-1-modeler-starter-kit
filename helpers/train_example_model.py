# Structure and code adapted from here: https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be

# This is a functional model in the sense that it "works", but it is not good by any metric.
# It is a starting point for an actual model
# The model is not trained on the entire dataset, only 10_000 to save time during the modelers first iteration

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import pandas as pd

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs



def train_example_model(training_dataframe, save_model=True, filepath=None, training_cols=None):

    # Create training and test sets

    X_train, X_test, y_train, y_test = train_test_split(training_dataframe[training_cols].to_numpy(),
                                                        training_dataframe['target'].to_numpy(), test_size=0.2)

    # Set parameters
    epochs = 100
    input_dim = X_train.shape[1]  # Number of features
    output_dim = 1  # Two possible target labels: 0 or 1
    learning_rate = 0.01
    losses = []
    losses_test = []
    Iterations = []
    iter = 0

    # Initialize model, loss, and optimizer
    model = LogisticRegression(input_dim, output_dim)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Convert to tensors
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)




    for epoch in tqdm(range(int(epochs)), desc='Training Epochs'):
        x = X_train
        labels = y_train
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(torch.squeeze(outputs), labels)  # [200,1] -squeeze-> [200]
        loss.backward()  # Computes the gradient of the given tensor w.r.t. graph leaves

        optimizer.step()  # Updates weights and biases with the optimizer (SGD)
        iter += 1
        if iter % epochs == 0:
            # calculate Accuracy
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)

                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test / total_test
                losses_test.append(loss_test.item())

                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct / total
                losses.append(loss.item())
                Iterations.append(iter)

                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    if save_model:
        torch.save(model.state_dict(), filepath)
    return model


if __name__ == '__main__':

    """
    This won't work in the updated starter kit, need to refactor!
    """

    borrow_training_dataframe = pd.read_parquet('../../data/borrow_training_dataframe.parquet')

    training_cols = list(borrow_training_dataframe.columns.drop(
        ['borrow_timestamp', 'wallet_address', 'borrow_block_number', 'target']))

    test_model = train_example_model(training_dataframe=borrow_training_dataframe,
                                     save_model=True, filepath='../../models/example_lr_model.pt',
                                     training_cols=training_cols)

    input_dim = len(training_cols)
    output_dim = 1
    model = LogisticRegression(input_dim,output_dim)
    model.load_state_dict(torch.load('../../models/example_lr_model.pt'))
    model.eval()

    input_for_prediction = borrow_training_dataframe.iloc[0][training_cols].astype(float).to_numpy()

    new_data = torch.tensor(input_for_prediction).float()
    with torch.no_grad():
        prediction = model(new_data)
        if prediction == 1.0:
            print(f'We predict this borrower will be liquidated')
        else:
            print(f'We predict this borrower will NOT be liquidated')