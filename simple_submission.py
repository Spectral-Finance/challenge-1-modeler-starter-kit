import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from tqdm.auto import tqdm
from timeit import default_timer as timer
import random
import os
import platform
from spectral_datawrappers.credit_scoring.credit_scoring_wrapper import CreditScoringWrapper
import configparser
from helpers.modeling import (TestData, TrainData, StratifiedBatchSampler,
                              auc_fn, auroc_fn, auc_pr_fn,
                              brier_fn, ks_fn, recall_fn, prob_diff_fn, f1_score_fn,
                              train_step, test_step, eval_model,
                              ValidationLossEarlyStopping)

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.determenistic = True
random.seed(random_seed)
g = torch.Generator()
g.manual_seed(random_seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(random_seed)



class PredictLiquidationsV1(nn.Module):
    """
    The final layer should be a sigmoid, to get the probability of liquidation.
    """

    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

def seed_worker():
    worker_seed = random_seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_simple_submission():
    """
    This function will download the training data, train a model, and submit the predictions.
    This is to demonstrate how you can create an example submission using the Spectral CLI.
    This will not result in a high score, but it will get you started.
    We encourage you to walk through the starter kit notebook and improve upon these results.
    """
    # Download Training Data
    print('Downloading training data')
    os.system("spectral-cli fetch-training-data 0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544")

    training_dataframe = duckdb.query((f"""
    select * from '{'0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544_training_data.parquet'}'
    where max_risk_factor < 100
    """)).df().drop(columns=['withdraw_amount_sum_eth'], inplace=False)


    training_cols = list(training_dataframe.columns.drop(
        ['borrow_timestamp', 'wallet_address', 'borrow_block_number', 'target',
         'withdraw_deposit_diff_If_positive_eth']))\


    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(training_dataframe[training_cols].to_numpy(),
                                                      training_dataframe['target'].to_numpy(),
                                                      test_size=0.2,
                                                      random_state=random_seed)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)

    train_data = TrainData(torch.from_numpy(X_train_scaled).type(torch.float),
                           torch.from_numpy(y_train).type(torch.float))

    validation_data = TestData(torch.from_numpy(X_val_scaled).type(torch.float),
                               torch.from_numpy(y_val).type(torch.float))

    NUM_WORKERS = 0  # use all available CPU cores with os.cpu_count(), if possible
    BATCH_SIZE = int(X_train.shape[0] / 100)  # ~1% of the training data

    # initialize DataLoaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_sampler=StratifiedBatchSampler(torch.tensor(y_train), batch_size=BATCH_SIZE),
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  num_workers=NUM_WORKERS)
    validation_dataloader = DataLoader(dataset=validation_data,
                                       batch_size=BATCH_SIZE,
                                       shuffle=False,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       num_workers=NUM_WORKERS)

    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    validation_features_batch, validation_labels_batch = next(iter(validation_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate the model
    model_1 = PredictLiquidationsV1(input_features=X_train.shape[1],
                                    output_features=1,
                                    hidden_units=82).to(device)

    # Initialize early stopping
    early_stopper = ValidationLossEarlyStopping(patience=1, min_delta=0.0)

    # Define loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(params=model_1.parameters(),
                           lr=0.001,
                           weight_decay=0.01)

    # Measure training time
    train_time_start = timer()

    # Set the number of training epochs
    # Note: You may want to increase the number of epochs for your final model
    epochs = 5

    # Create empty lists to track loss values
    model_1_train_loss_values = []
    model_1_validation_loss_values = []
    model_1_epoch_count = []

    # Training loop
    print('Training model')
    for epoch in tqdm(range(epochs)):
        train_metrics = train_step(data_loader=train_dataloader,
                                   model=model_1,
                                   epochs=epoch,
                                   loss_fn=loss_fn,
                                   optimizer=optimizer,
                                   auc_fn=auc_fn,
                                   recall_fn=recall_fn,
                                   f1_score_fn=f1_score_fn,
                                   auroc_fn=auroc_fn,
                                   brier_fn=brier_fn,
                                   auc_pr_fn=auc_pr_fn,
                                   ks_fn=ks_fn,
                                   prob_diff_fn=prob_diff_fn,
                                   simple_submission=True
                                   )
        validation_metrics = test_step(data_loader=validation_dataloader,
                                       model=model_1,
                                       epochs=epoch,
                                       loss_fn=loss_fn,
                                       auc_fn=auc_fn,
                                       recall_fn=recall_fn,
                                       f1_score_fn=f1_score_fn,
                                       auroc_fn=auroc_fn,
                                       brier_fn=brier_fn,
                                       auc_pr_fn=auc_pr_fn,
                                       ks_fn=ks_fn,
                                       prob_diff_fn=prob_diff_fn,
                                       simple_submission=True
                                       )
        model_1_epoch_count.append(epoch)
        model_1_train_loss_values.append(train_metrics['loss'])
        model_1_validation_loss_values.append(validation_metrics['loss'])

        if early_stopper.early_stop_check(validation_metrics['loss']):
            print(f"Stopped early at epoch: {epoch}")
            break

    # Calculate model 1 results
    model_1_results = eval_model(model=model_1,
                                 data_loader=validation_dataloader,
                                 loss_fn=loss_fn,
                                 auc_fn=auc_fn,
                                 recall_fn=recall_fn,
                                 f1_score_fn=f1_score_fn,
                                 auroc_fn=auroc_fn,
                                 brier_fn=brier_fn,
                                 auc_pr_fn=auc_pr_fn,
                                 ks_fn=ks_fn,
                                 prob_diff_fn=prob_diff_fn,
                                 device=device)
    model_1_results.update({'model_name': model_1.__class__.__name__})

    if not os.path.exists('submissions'):
        os.makedirs('submissions')

    # Export the ONNX file using the trained model and create an observation for model calibration.
    model_1.eval()
    torch.onnx.export(model_1,
                      torch.randn((1, X_train.shape[1]), requires_grad=True).to(device),
                      'submissions/model_1.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    x = validation_features_batch[:1].reshape([-1]).numpy().tolist()
    data = dict(input_data=[x])
    json.dump(data, open('submissions/model_1_calibration.json', 'w'))


    # Zane uncomment when ready to commit
    # os.system("spectral-cli commit submissions/model_1.onnx submissions/model_1_calibration.json 0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544")
    print('Committed model, waiting 30 seconds for test set')
    # time.sleep(30)

    os.system("spectral-cli fetch-testing-data 0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544")

    test_set_addresses = pd.read_parquet('0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544_testing_dataset.parquet')[
        'wallet_address'].tolist()

    print('Fetching features for test set, this will take several minutes')

    if platform.system() != 'windows':
        config = configparser.ConfigParser()
        config.read_file(open(os.path.expanduser("~/.spectral/config.ini")))
    else:
        config = configparser.ConfigParser()
        config_path = os.path.join(os.environ['USERPROFILE'], '.spectral', 'config.ini')
        config.read(config_path)

    # Read our key and set a client
    spectral_api_key = config['global']['spectral_api_key']
    client = CreditScoringWrapper({'spectral_api_key': spectral_api_key})

    response = client.request_batch(test_set_addresses)
    test_set_with_features = pd.DataFrame(response)

    testing_samples_scaled = sc.transform(test_set_with_features[training_cols])

    model_1.eval()

    # Inference using the trained model
    with torch.inference_mode():
        pol = model_1(torch.tensor(testing_samples_scaled).float().to(device)).squeeze().detach().cpu().numpy()

    # Create a DataFrame with predictions
    submission_dataframe = pd.DataFrame(testing_samples_scaled, index=test_set_with_features.index,
                                        columns=test_set_with_features[training_cols].columns)
    submission_dataframe['wallet_address'] = test_set_with_features['wallet_address']
    submission_dataframe['pred_prob'] = pol

    submission_dataframe['pred_label'] = (
        submission_dataframe['pred_prob'].apply(lambda x: 1 if x > .6 else 0))

    # Format the DataFrame for submission
    non_feature_cols = ['wallet_address', 'pred_prob', 'pred_label']

    # Rename feature columns for anonymity (optional)
    for index, col in enumerate(submission_dataframe.columns):
        if col not in non_feature_cols:
            submission_dataframe.rename(columns={col: f'feature_{index + 1}'}, inplace=True)

    # Order the columns for readability (optional)
    cols_order_list = non_feature_cols + [col for col in submission_dataframe.columns if col not in non_feature_cols]
    submission_dataframe = submission_dataframe[cols_order_list]

    # Save the submission file
    submission_dataframe.to_parquet('submissions/submission.parquet', index=False)

    # Zane uncomment when ready to submit
    # os.system("spectral-cli submit-inferences 0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544 submissions/submission.parquet")
    print('submitted!')
if __name__ == '__main__':
    create_simple_submission()
