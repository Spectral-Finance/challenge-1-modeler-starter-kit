import pandas as pd
import onnx
import numpy as np
import joblib
import onnxruntime
from spectral_datawrappers.credit_scoring.credit_scoring_wrapper import CreditScoringWrapper
import os
import configparser
import platform




def make_example_prediction_for_address(wallet_address: str) -> float:
    """
    Given a wallet address, return a prediction of the probability of liquidation
    based on the model we trained in the starter kit
    :param wallet_address: A wallet address to make a prediction for
    :return probability_of_liquidation: A float representing the probability of liquidation
    """

    model_onnx_filepath = 'submissions/model_1.onnx'

    # Read your Spectral api_key from your config file

    # Macos and Linux users
    if platform.system() != 'Windows':
        config = configparser.ConfigParser()
        config.read_file(open(os.path.expanduser("~/.spectral/config.ini")))
    else:
        config = configparser.ConfigParser()
        config_path = os.path.join(os.environ['USERPROFILE'], '.spectral', 'config.ini')
        config.read(config_path)

    # Read our key and set a client
    spectral_api_key = config['global']['spectral_api_key']
    client = CreditScoringWrapper({'spectral_api_key': spectral_api_key})

    test_request = client.request({"wallet_address": wallet_address})
    training_cols = list(test_request.keys())
    training_cols.remove('wallet_address')
    training_cols.remove('withdraw_amount_sum_eth')
    model_input_dataframe = pd.DataFrame(test_request, index=[0])

    sc = joblib.load('model/example_standard_scaler.pkl')  # load our scaler
    # Scale the test data in the same way we scaled our training and validation data
    test_input_scaled = sc.transform(model_input_dataframe[training_cols].to_numpy())

    # load the model version
    # generate the ML model output from the ONNX file
    onnx_model = onnx.load(model_onnx_filepath)
    onnx.checker.check_model(onnx_model)
    inputs_onnx = np.array(test_input_scaled).astype(np.float32)
    onnx_session = onnxruntime.InferenceSession(model_onnx_filepath)
    onnx_input = {onnx_session.get_inputs()[0].name: inputs_onnx}
    onnx_output_prob = onnx_session.run(None, onnx_input)
    probability_of_liquidation = abs(onnx_output_prob[0][0][0])
    return probability_of_liquidation


if __name__ == '__main__':
    # Example usage
    wallet_address = '0x9d9c3513189342c8e24a987fd25df3bda68b2af4'
    probability_of_liquidation_prediction = make_example_prediction_for_address(wallet_address)
    print(f'The predicted probability of liquidation for {wallet_address} in the next 30 days is {round(probability_of_liquidation_prediction, 4)}')