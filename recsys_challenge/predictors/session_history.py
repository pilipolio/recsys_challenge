import os

import numpy as np
import pandas as pd

from recsys_challenge.base import Session, evaluate, validation_dataset
from recsys_challenge.predictors import item_pop


def predict_sessions(test_clicks, item_stats, n_clicks_threshold, rate_threshold):
    predicted_test_clicks = test_clicks.merge(
        on='ITEM_ID',
        right=item_stats[['BUYS/SESSIONS']].reset_index()
    )
    predicted_mask = np.logical_and(
        predicted_test_clicks.N_CLICKS >= n_clicks_threshold,
        predicted_test_clicks['BUYS/SESSIONS'] > rate_threshold)
        
    predicted_test_clicks = predicted_test_clicks.ix[
        predicted_mask, ['SESSION_ID', 'ITEM_ID']]
    return Session.group_sessions(
        predicted_test_clicks.values)


if __name__ == '__main__':
    data_directory = 'data'
    clicks_and_buys = pd.read_pickle(os.path.join(data_directory, 'clicks_and_buys.df'))
    test_clicks = pd.read_pickle(os.path.join(data_directory, 'test_clicks.df'))

    item_stats = item_pop.item_statistics(clicks_and_buys)

    validation_clicks_and_buys, validation_sessions = validation_dataset(clicks_and_buys, size=1000000)

    import itertools
    for n, r in itertools.product([2, 3], [.01, .025, .05, .075, .1]):
        print n, r
        predicted_validation_sessions = predict_sessions(validation_clicks_and_buys, item_stats, n_clicks_threshold=n, rate_threshold=r)

        print evaluate(predicted_validation_sessions, validation_sessions)
    
    predicted_test_sessions = predict_sessions(test_clicks, item_stats, n_clicks_threshold=2, rate_threshold=0.05)

    # 2clicks_and_p5 => ~25000
    # 2clicks_or_p5 was better on the validation set but only ~15000 on the test
    with open('./solutions/session_history_2clicks_and_p5_solution.dat', 'w') as fp:
        fp.writelines(session.to_csv_line() for session in predicted_test_sessions)
