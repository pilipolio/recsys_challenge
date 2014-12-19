import os

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder
from sklearn.linear_model import SGDClassifier

from recsys_challenge.base import Session, evaluate, validation_dataset

class EncodedModel(object):
    def __init__(self, model, range_encoder, category_encoder):
        self.model = model
        self.range_encoder = range_encoder
        self.category_encoder = category_encoder

    def predict(self, X):
        compressed_ids = self.range_encoder.transform(X[:,0])
        X[:,0] = compressed_ids
        X_encoded = self.category_encoder.transform(X)
        return self.model.predict(X_encoded)
    
    def decision_function(self, X):
        compressed_ids = self.range_encoder.transform(X[:,0])
        X[:,0] = compressed_ids
        X_encoded = self.category_encoder.transform(X)
        print X_encoded.shape
        return self.model.decision_function(X_encoded)

    
def train_logistic_model(clicks_and_buys, **params):

    compressed_range_encoder = LabelEncoder()
    compressed_item_ids = compressed_range_encoder.fit_transform(clicks_and_buys.ITEM_ID)

    category_encoder = OneHotEncoder(dtype=np.int, categorical_features=[0])
    X = np.column_stack((
        compressed_item_ids,
        clicks_and_buys.N_CLICKS,
        scale((clicks_and_buys.SESSION_STOP - clicks_and_buys.ITEM_STOP).astype('timedelta64[s]')),
        scale((clicks_and_buys.SESSION_STOP - clicks_and_buys.SESSION_START).astype('timedelta64[s]')),
        scale((clicks_and_buys.ITEM_STOP - clicks_and_buys.ITEM_START).astype('timedelta64[s]')),
    ))
    
    X_encoded = category_encoder.fit_transform(X)
    
    print X.shape

    logr = SGDClassifier(loss='log', fit_intercept=False, **params)
    logr.fit(
        X=X_encoded,
        y=clicks_and_buys.N_BUYS > 0)
    return EncodedModel(logr, compressed_range_encoder, category_encoder)

    
def logistic_predict(fitted_model, test_clicks):
    X_test = np.column_stack((
        test_clicks.ITEM_ID,
        test_clicks.N_CLICKS,
        scale((test_clicks.SESSION_STOP - test_clicks.ITEM_STOP).astype('timedelta64[s]')),
        scale((test_clicks.SESSION_STOP - test_clicks.SESSION_START).astype('timedelta64[s]')),
        scale((test_clicks.ITEM_STOP - test_clicks.ITEM_START).astype('timedelta64[s]')),
    ))

    p_ys = fitted_model.decision_function(X=X_test)
    p_threshold = np.percentile(p_ys, 100 * (1 - 0.05))
    return test_clicks[p_ys >= p_threshold]


def predict_sessions(fitted_model, test_clicks):
    predicted_buys = logistic_predict(fitted_model, test_clicks)
    return Session.group_sessions(
        predicted_buys[['SESSION_ID', 'ITEM_ID']].values)


if __name__ == '__main__':
    data_directory = 'data'
    clicks_and_buys = pd.read_pickle(os.path.join(data_directory, 'clicks_and_buys.df'))

    test_clicks = pd.read_pickle(os.path.join(data_directory, 'test_clicks.df'))

    print 'split validation set'
    validation_clicks_and_buys, validation_sessions = validation_dataset(clicks_and_buys, size=1000000)
    print validation_clicks_and_buys.shape

    print 'train on clicks_and_buys {}'.format(clicks_and_buys.shape)

    for alpha in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
        model = train_logistic_model(clicks_and_buys, alpha=alpha)
        print model.model
        #print np.sum(np.power(model.model.coef_, 2))
        print 'predict on validation_clicks_and_buys {}'.format(validation_clicks_and_buys.shape)
        predicted_validation_sessions = predict_sessions(model, validation_clicks_and_buys)
        print evaluate(predicted_validation_sessions, validation_sessions)
