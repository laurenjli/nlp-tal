'''
Just some utility fns to be used across notebooks
'''
from ast import literal_eval
import pandas as pd


def load_and_clean(file='total_trans_normalized.csv', grouped=True):
    '''
    Loads in and cleans our tokenized transcript csv

    Will group by episode if noted
    '''
    convert_dict = {
        'tokenized_sents': literal_eval,
        'no_lemma_normalized_sents': literal_eval,
        'normalized_sents':literal_eval,
        'tokenized_text':literal_eval,
        'normalized_tokens':literal_eval, 
        'no_lemma_normalized_tokens':literal_eval

    }
    tal_df = pd.read_csv(file, converters=convert_dict)

    group_cols = {
        'ep_num': 'max', 'url': 'max',
    'text': 'sum', 'tokenized_sents': 'sum',
    'no_lemma_normalized_sents': 'sum',
    'normalized_sents': 'sum',
    'tokenized_text': 'sum',
    'normalized_tokens': 'sum',
    'no_lemma_normalized_tokens': 'sum',
    'year': 'max'
    }

    if grouped:
        tal_df = tal_df.groupby('ep_title').agg(group_cols).reset_index()
    return tal_df