import os

import pandas as pd
import numpy as np


BUYS_COLUMNS = ['SESSION_ID', 'TS', 'ITEM_ID', 'PRICE', 'Q']
CLICKS_COLUMNS = ['SESSION_ID', 'TS', 'ITEM_ID', 'CATEGORY']


def time_collapsed_clicks(clicks):
    """
    >>> clicks_columns = ['SESSION_ID', 'TS', 'ITEM_ID', 'CATEGORY']
    >>> clicks_logs = [[1, 't0', 10, 0], [1, 't1', 20, 0], [1, 't2', 20, 0]]
    >>> clicks = pd.DataFrame(columns=clicks_columns, data=clicks_logs)

    >>> time_collapsed_clicks(clicks)
       SESSION_ID  ITEM_ID  CATEGORY ITEM_START ITEM_STOP  N_CLICKS SESSION_START SESSION_STOP
    0           1       10         0         t0        t0         1            t0           t2
    1           1       20         0         t1        t2         2            t0           t2
    """

    clicks_by_sessions = clicks.groupby(['SESSION_ID'])
    session_limits = clicks_by_sessions.aggregate({'TS': [np.min, np.max]})
    session_limits.columns = ['SESSION_START', 'SESSION_STOP']
    
    clicks_by_item_and_sessions = clicks.groupby(['SESSION_ID', 'ITEM_ID'])
    item_sub_sessions = clicks_by_item_and_sessions.aggregate({
        'TS': [np.min, np.max],
        'CATEGORY': np.min,
        'SESSION_ID': len
    })
    # kind of fragile as based on column names alphabetical orders as dict keys?
    item_sub_sessions.columns = ['CATEGORY', 'ITEM_START', 'ITEM_STOP', 'N_CLICKS']
    
    collapsed_clicks = pd.merge(
        left=item_sub_sessions.reset_index(),
        right=session_limits.reset_index(),
        on='SESSION_ID',
        how='inner')

    return collapsed_clicks

    
def collapse_time_and_join(buys, clicks):
    """ Collapsing the time dimension by grouping by (session_id, item_id)
    and then join clicks and buys rows.
    """
    buys = buys.groupby(['SESSION_ID', 'ITEM_ID']).size().reset_index()
    buys.columns = ['SESSION_ID', 'ITEM_ID', 'N_BUYS']

    clicks = time_collapsed_clicks(clicks)
    
    clicks_and_buys = buys.merge(
        right=clicks, on=['SESSION_ID', 'ITEM_ID'],
        how='outer')
    # fills N_CLICKS and N_BUYS to 0 when missing
    clicks_and_buys = clicks_and_buys.fillna(0)
    # for some reasone SESSION_ID and ITEM_ID get also converted to float by the merge
    clicks_and_buys[['N_CLICKS', 'N_BUYS']] = clicks_and_buys[['N_CLICKS', 'N_BUYS']].astype(np.int64)
    clicks_and_buys[['SESSION_ID', 'ITEM_ID']] = clicks_and_buys[['SESSION_ID', 'ITEM_ID']].astype(np.int64)
    
    return clicks_and_buys


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.iteritems():
        df.to_pickle(os.path.join(data_directory, name + '.df'))
        
    
if __name__ == '__main__':
    data_directory = 'data'

    buys = pd.read_csv(os.path.join(data_directory, 'yoochoose-buys.dat'), names=BUYS_COLUMNS, parse_dates=['TS'])
    clicks = pd.read_csv(os.path.join(data_directory, 'yoochoose-clicks.dat'), names=CLICKS_COLUMNS, parse_dates=['TS'])
    clicks_and_buys = collapse_time_and_join(buys, clicks)
    to_pickled_df(data_directory, clicks_and_buys=clicks_and_buys)
    
    test_clicks = pd.read_csv(os.path.join(data_directory, 'yoochoose-test.dat'), names=CLICKS_COLUMNS)
    test_clicks = time_collapsed_clicks(test_clicks)

    to_pickled_df(data_directory, test_clicks=test_clicks)
