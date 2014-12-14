import os

import numpy as np
import pandas as pd

from recsys_challenge.base import Session, evaluate, validation_dataset


def item_statistics(clicks_and_buys):
    
    def n_unique_sessions(session_ids):
        return len(np.unique(session_ids))

    buys_clicks_by_items = pd.DataFrame.from_dict({
        'N_CLICKS': clicks_and_buys[clicks_and_buys.N_CLICKS > 0].groupby('ITEM_ID').size(),
        'N_SESSIONS': clicks_and_buys[clicks_and_buys.N_CLICKS > 0].groupby('ITEM_ID')['SESSION_ID'].agg(n_unique_sessions),
        'N_BUYS': clicks_and_buys[clicks_and_buys.N_BUYS > 0].groupby('ITEM_ID').size()
    }).fillna(0)

    buys_clicks_by_items.index.name = 'ITEM_ID'
    buys_clicks_by_items['BUYS/SESSIONS'] = buys_clicks_by_items['N_BUYS'] / buys_clicks_by_items['N_SESSIONS']
    buys_clicks_by_items = buys_clicks_by_items.sort('N_BUYS', ascending=False)
    
    return buys_clicks_by_items

def predict_sessions(clicks_and_buys, item_stats, rate_threshold):
    predicted_buys = clicks_and_buys.merge(
        on='ITEM_ID',
        right=item_stats['BUYS/SESSIONS'].reset_index()
    ).sort('SESSION_ID')
    predicted_buys = predicted_buys[predicted_buys['BUYS/SESSIONS'] > rate_threshold]

    return Session.group_sessions(
        predicted_buys[['SESSION_ID', 'ITEM_ID']].values)

    
if __name__ == '__main__':
    data_directory = 'data'
    clicks_and_buys = pd.read_pickle(os.path.join(data_directory, 'clicks_and_buys.df'))
    test_clicks = pd.read_pickle(os.path.join(data_directory, 'test_clicks.df'))

    validation_clicks_and_buys, validation_sessions = validation_dataset(clicks_and_buys, size=1000000)

    item_stats = item_statistics(clicks_and_buys)

    # debug
    print item_stats.sort('N_BUYS', ascending=False).head(10)
    import matplotlib.pyplot as plt
    plt.hist(item_stats.ix[item_stats['N_BUYS'] > 0, 'BUYS/SESSIONS'], bins=50)
    plt.savefig('most_likely_bought_when_seen_item.png')
    plt.close()

    # VALIDATION
    predicted_validation_sessions = predict_sessions(validation_clicks_and_buys, item_stats, rate_threshold=.1)
    print evaluate(predicted_validation_sessions, validation_sessions)

    # Gives a buy/click ratio of .0525 and ~11,000 on http://2015.recsyschallenge.com/submission.html
    predicted_test_sessions = predict_sessions(test_clicks, item_stats, rate_threshold=.1)

    with open('./solutions/most_likely_bought_when_seen_solution.dat', 'w') as fp:
        fp.writelines(session.to_csv_line() for session in predicted_test_sessions)
