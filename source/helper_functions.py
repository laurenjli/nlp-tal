'''
This file contains helper functions for our work in the main jupyter notebook files.

Created: 2/23/2020
'''

# import functions
import pandas as pd 
from ast import literal_eval
from copy import deepcopy

# FILE LOADING AND MANIPULATION

# GLOBAL CONVERTER 
converters = {'tokenized_sents': literal_eval,'no_lemma_normalized_sents': literal_eval,
                'normalized_sents':literal_eval,'tokenized_text':literal_eval,
                'normalized_tokens':literal_eval,'no_lemma_normalized_tokens':literal_eval}

def agg_text(df, group_col):
    '''
    This function aggregates text in the tal dataframe for all text columns
    input: df (dataframe), group_col (name of grouping column)
    output: dataframe with text aggregated
    '''
    return df.groupby(group_col).agg({'text':'sum','tokenized_sents':'sum', 'no_lemma_normalized_sents': 'sum',
                                      'normalized_sents':'sum','tokenized_text':'sum',
                                      'normalized_tokens':'sum','no_lemma_normalized_tokens':'sum'}).reset_index()

def group(row):
    '''
    This function can be used to create a 'grouping' column in the tal dataframe based on year.
    input: row of a df
    output: integer of group

    Sample call: tal_df['group] = tal_df.apply(lambda x: group(x), axis=1)
    '''
    if row['year'] < 1998:
        return 1
    elif row['year'] < 2000:
        return 2
    elif row['year'] < 2002:
        return 3
    elif row['year'] < 2004:
        return 4
    elif row['year'] < 2006:
        return 5
    elif row['year'] < 2009:
        return 6
    elif row['year'] < 2011:
        return 7
    elif row['year'] < 2013:
        return 8
    elif row['year'] < 2015:
        return 9
    elif row['year'] < 2017:
        return 10
    else:
        return 11



# EMBEDDING DIMENSIONS

def normalize(vector):
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector

def dimension(model, positives, negatives):
    diff = sum([normalize(model[x]) for x in positives]) - sum([normalize(model[y]) for y in negatives])
    return diff

def clean_words(model,word_list):
    dupe = deepcopy(word_list)
    for w in dupe:
        try:
            model[w]
        except:
            dupe.remove(w)
    return dupe

def makeDF(model, word_list, dim_dict):
    new_dict = {}
    for k,v in dim_dict.items():
        tmp = []
        for word in word_list:
            tmp.append(sklearn.metrics.pairwise.cosine_similarity(model[word].reshape(1,-1), v.reshape(1,-1))[0][0])
        new_dict[k] = tmp
    df = pd.DataFrame(new_dict, index = word_list)
    return df

def Coloring(Series):
    x = Series.values
    y = x-x.min()
    z = y/y.max()
    c = list(plt.cm.rainbow(z))
    return c

def PlotDimension(ax,df, dim):
    ax.set_frame_on(False)
    ax.set_title(dim, fontsize = 20)
    colors = Coloring(df[dim])
    for i, word in enumerate(df.index):
        ax.annotate(word, (0, df[dim][i]), color = colors[i], alpha = 0.6, fontsize = 12)
    MaxY = df[dim].max()
    MinY = df[dim].min()
    plt.ylim(MinY,MaxY)
    plt.yticks(())
    plt.xticks(())