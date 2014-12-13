from itertools import imap
from collections import defaultdict, namedtuple

import numpy as np


class Session(namedtuple('Session', ['id', 'item_ids'])):

    @property
    def has_bought(self):
        return len(self.item_ids) > 0
    
    @staticmethod
    def group_sessions(session_and_item_ids):
        """ Returns a sequence of Session by ids given (session_id, item_id) tuples
        """
        item_ids_by_sessions = defaultdict(list)

        for session_id, item_id in session_and_item_ids:
            item_ids_by_sessions[session_id].append(item_id)
        return [Session(session_id, set(item_ids))
                for session_id, item_ids in item_ids_by_sessions.iteritems()]

    @staticmethod
    def group_validation_sessions(session_and_item_and_has_boughts):
        """ Returns a sequence of Session by ids given (session_id, item_id, has_bought) tuples
        >>> list(Session.group_validation_sessions([('S_1', 'I_1', 0), ('S_2', 'I_1', 0), ('S_1', 'I_2', 1)]))
        [Session(id='S_1', item_ids=set(['I_2'])), Session(id='S_2', item_ids=set([]))]
        """
        item_ids_by_sessions = dict()

        for session_id, item_id, has_bought in session_and_item_and_has_boughts:
            if session_id not in item_ids_by_sessions:
                item_ids_by_sessions[session_id] = []
            if has_bought:
                item_ids_by_sessions[session_id].append(item_id)

        return [Session(session_id, set(item_ids))
                for session_id, item_ids in item_ids_by_sessions.iteritems()]

    def to_csv_line(self):
        return '{session_id};{item_ids_csv}\n'.format(
            session_id=self.id, item_ids_csv=','.join(map(str, self.item_ids)))

# http://2015.recsyschallenge.com/challenge.html
EvaluationMeasure = namedtuple(
    'EvaluationMeasure',
    ['n_predicted_sessions', 'n_predicted_items', 'n_ever_predicted_items', 'n_hits', 'n_misses', 'sum_jaccard_scores', 'buying_session_ratio', 'total'])

from itertools import chain


def evaluate(predicted_sessions, test_sessions):
    n_hits, n_misses = prediction_hits_misses(predicted_sessions, test_sessions)
    sum_scores = sum_jaccard_scores(predicted_sessions, test_sessions)
    ratio = buying_session_ratio(test_sessions)
    
    return EvaluationMeasure(
        len(predicted_sessions), sum(len(s.item_ids) for s in predicted_sessions), len(set(chain(*[s.item_ids for s in predicted_sessions]))),
        n_hits, n_misses,
        sum_scores,
        ratio,
        total=sum_scores + ratio * (n_hits - n_misses)
    )

    
def prediction_hits_misses(predicted_sessions, test_sessions):
    """  
    >>> prediction_hits_misses([Session(1, {}), Session(2, {}), Session(3, {})], test_sessions=[Session(1, {'ITEM'}), Session(2, {})])
    (1, 2)
    """
    pred = set(p.id for p in predicted_sessions)
    test = set(t.id for t in test_sessions if len(t.item_ids) > 0)
    return len(pred & test), len(pred - test)


def sum_jaccard_scores(predicted_sessions, test_sessions):
    """  
    >>> sum_jaccard_scores([Session(1, {'ITEM1'}), Session(2, {'ITEM1'})], test_sessions=[Session(1, {'ITEM1'}), Session(2, {'ITEM2'})])
    1.0
    """
    test_sessions_by_ids = {s.id:s for s in test_sessions}
    hit_pred_and_test_sessions = ((p, test_sessions_by_ids[p.id]) for p in predicted_sessions if p.id in test_sessions_by_ids)
    return sum(jaccard_score(p, t) for p, t in hit_pred_and_test_sessions)


def jaccard_score(predicted_session, test_session):
    """
    >>> jaccard_score(Session(1, set()), Session(1, {'ITEM1'}))
    0.0
    >>> jaccard_score(Session(1, {'ITEM1'}), Session(1, {'ITEM1'}))
    1.0
    >>> jaccard_score(Session(1, {'ITEM1', 'ITEM2'}), Session(1, {'ITEM1'}))
    0.5
    """
    return len(predicted_session.item_ids & test_session.item_ids) / float(len(predicted_session.item_ids | test_session.item_ids))


def buying_session_ratio(test_sessions):
    """
    >>> buying_session_ratio([Session(1, set()), Session(2, {'ITEM_BOUGHT'}), Session(3, {}), Session(4, set())])
    0.25
    """
    return len([t for t in test_sessions if t.has_bought]) / float(len(test_sessions))


def validation_dataset(clicks_and_buys):
    validation_indexes = np.random.permutation(np.arange(clicks_and_buys.shape[0]))[:1000000]
    validation_clicks_and_buys = clicks_and_buys.iloc[validation_indexes,:]
    validation_sessions = Session.group_validation_sessions(
        validation_clicks_and_buys[['SESSION_ID', 'ITEM_ID', 'N_BUYS']].values)
    return validation_clicks_and_buys, validation_sessions
