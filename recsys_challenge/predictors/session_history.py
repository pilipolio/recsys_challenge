import os

import numpy as np
import pandas as pd

from recsys_challenge.base import Session, evaluate, validation_dataset


def predict_sessions(test_clicks, n_clicks_threshold):
    predicted_test_clicks = test_clicks[test_clicks.N_CLICKS >= n_clicks_threshold]
    return Session.group_sessions(
        predicted_test_clicks[['SESSION_ID', 'ITEM_ID']].values)


if __name__ == '__main__':
    data_directory = 'data'
    clicks_and_buys = pd.read_pickle(os.path.join(data_directory, 'clicks_and_buys.df'))
    test_clicks = pd.read_pickle(os.path.join(data_directory, 'test_clicks.df'))

    validation_clicks_and_buys, validation_sessions = validation_dataset(clicks_and_buys)

    predicted_validation_sessions = predict_sessions(validation_clicks_and_buys, n_clicks_threshold=2)

    print evaluate(predicted_validation_sessions, validation_sessions)

    predicted_test_sessions = predict_sessions(test_clicks, n_clicks_threshold=2)
        
    with open('./solutions/test_items_with_3clicks_solution.dat', 'w') as fp:
        fp.writelines(session.to_csv_line() for session in predicted_test_sessions)
