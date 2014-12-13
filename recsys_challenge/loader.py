import pandas as pd
import os

def from_csv_in_directory(data_directory):
    buys = pd.read_csv(os.path.join(data_directory, 'yoochoose-buys.dat'), names=['SESSION_ID', 'TS', 'ITEM_ID', 'PRICE', 'Q'])
    clicks = pd.read_csv(os.path.join(data_directory, 'yoochoose-clicks.dat'), names=['SESSION_ID', 'TS', 'ITEM_ID', 'CATEGORY'])
    test_clicks = pd.read_csv(os.path.join(data_directory, 'yoochoose-test.dat'), names=['SESSION_ID', 'TS', 'ITEM_ID', 'CATEGORY'])
    return buys, clicks, test_clicks


def to_time_collapsed(buys, clicks, test_clicks):
    """ Collapsing the time dimension by grouping by (session_id, item_id)
    and then join clicks and buys rows.
    """
    buys = buys.groupby(['SESSION_ID', 'ITEM_ID']).size().reset_index()
    buys.columns = ['SESSION_ID', 'ITEM_ID', 'N_BUYS']

    clicks = clicks.groupby(['SESSION_ID', 'ITEM_ID', 'CATEGORY']).size().reset_index()
    clicks.columns = ['SESSION_ID', 'ITEM_ID', 'CATEGORY', 'N_CLICKS']

    test_clicks = test_clicks.groupby(['SESSION_ID', 'ITEM_ID', 'CATEGORY']).size().reset_index()
    test_clicks.columns = ['SESSION_ID', 'ITEM_ID', 'CATEGORY', 'N_CLICKS']

    clicks_and_buys = buys.merge(
        right=clicks, on=['SESSION_ID', 'ITEM_ID'],
        how='outer',
        suffixes=('_BUY', '_CLICKED')).sort('SESSION_ID').fillna(0)

    return clicks_and_buys, test_clicks


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.iteritems():
        df.to_pickle(os.path.join(data_directory, name + '.df'))
        
    
if __name__ == '__main__':
    data_directory = '/home/guillaume/Documents/recsys_challenge/yoochoose-data'
    buys, clicks, test_clicks = from_csv_in_directory(data_directory)
    clicks_and_buys, test_clicks = to_time_collapsed(buys, clicks, test_clicks)
    to_pickled_df(data_directory, clicks_and_buys=clicks_and_buys, test_clicks=test_clicks)
