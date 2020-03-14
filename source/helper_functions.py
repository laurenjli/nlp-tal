'''
This file contains helper functions for our work in the main jupyter notebook files.

Created: 2/23/2020
'''

# import functions
import pandas as pd 
import numpy as np
import gensim
from ast import literal_eval
import re
import copy
import nltk
import seaborn as sns
import spacy
import sklearn
import matplotlib.pyplot as plt
import networkx as nx


# FILE LOADING AND MANIPULATION

# GLOBAL CONVERTER 
converters = {'tokenized_sents': literal_eval,'no_lemma_normalized_sents': literal_eval,
                'normalized_sents':literal_eval,'tokenized_text':literal_eval,
                'normalized_tokens':literal_eval,'no_lemma_normalized_tokens':literal_eval}

def load_df(file, convert_dict=converters):
    '''
    Loads in TAL transcript file & outputs pd df w/ correct types
    '''
    return pd.read_csv(file, converters=convert_dict)


def agg_text(df, group_col, include_ep_info=False):
    '''
    This function aggregates text in the tal dataframe for all text columns
    input: df (dataframe), group_col (name of grouping column)
    output: dataframe with text aggregated
    '''
    if include_ep_info:
        agg_dict = {'ep_num': 'max', 'year': 'max', 'url': 'max', 
                                      'text':'sum','tokenized_sents':'sum', 'no_lemma_normalized_sents': 'sum',
                                      'normalized_sents':'sum','tokenized_text':'sum',
                                      'normalized_tokens':'sum','no_lemma_normalized_tokens':'sum'}
    else:
        agg_dict = {'text':'sum','tokenized_sents':'sum', 'no_lemma_normalized_sents': 'sum',
                                      'normalized_sents':'sum','tokenized_text':'sum',
                                      'normalized_tokens':'sum','no_lemma_normalized_tokens':'sum'}
    return df.groupby(group_col).agg(agg_dict).reset_index()

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


def split_five_years(tal_df):
    '''
    Adds a col to a df with "time period" int.
    Where time periods are defined as follows:
        1995-2001 (technically 6 years but so few eps in 1995)
        2002-2007
        2008-2013
        2014-2019

    Inputs:
        tal_df: a pandas df

    Outputs: an updated pandas df with add'l col of time period
    '''
    # Set groups
    group_1 = list(range(1995, 2002))
    group_2 = list(range(2002, 2008))
    group_3 = list(range(2008, 2014))
    group_4 = list(range(2014, 2020))
    yr_groups = [group_1, group_2, group_3, group_4]
    # Initialize col
    tal_df['five_yr_group'] = np.nan
    # Update col with group number (0-3)
    for idx, group in enumerate(yr_groups):
        tal_df.loc[tal_df['year'].isin(group), 'five_yr_group'] = int(idx)
    return tal_df

def clean_raw_text(raw_text, df, year):
    '''
    Using this to clean the COCA news texts
    '''
    articles = raw_text.split("##")
    for article in articles:
        if len(article) > 7:
            id_num = article[:7]
            text = article[7:]
            clean_text = replace_text(text)
            df.loc[len(df)] = [year, id_num, clean_text]


def replace_text(text):
    return text.replace("<p>", "").replace("\'m", "'m").replace("\'ll", "'ll"). \
                      replace("\'re", "'re").replace("\'s", "'s"). \
                      replace("\'re", "'re").replace("n\'t", "n't"). \
                      replace("\'ve", "'ve").replace("\'d", "'d"). \
                      replace("@ ", "")


# EMBEDDING DIMENSIONS

def normalize(vector):
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector

def dimension(model, positives, negatives):
    diff = sum([normalize(model[x]) for x in positives]) - sum([normalize(model[y]) for y in negatives])
    return diff

def clean_words(model,word_list):
    dupe = copy.deepcopy(word_list)
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

# LINGUISTIC CHANGE

def calc_syn0norm(model):
    """since syn0norm is now depricated"""
    return (model.wv.syn0 / np.sqrt((model.wv.syn0 ** 2).sum(-1))[..., np.newaxis]).astype(np.float32)

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
    (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """
    base_embed = copy.copy(base_embed)
    other_embed = copy.copy(other_embed)
    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = calc_syn0norm(in_base_embed)
    other_vecs = calc_syn0norm(in_other_embed)

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.syn0norm = other_embed.wv.syn0 = (calc_syn0norm(other_embed)).dot(ortho)
    return other_embed
    
def intersection_align_gensim(m1,m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1&vocab_m2
    if words: common_vocab&=set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

    # Then for each model...
    for m in [m1,m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = calc_syn0norm(m)
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index,word in enumerate(common_vocab):
            old_vocab_obj=old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return (m1,m2)


def findDiverence(word, embeddingsDict):
    cats = sorted(set(embeddingsDict.keys()))
    
    dists = []
    for embed in embeddingsDict[cats[0]][1:]:
        dists.append(1 - sklearn.metrics.pairwise.cosine_similarity(np.expand_dims(embeddingsDict[cats[0]][0][word], axis = 0), np.expand_dims(embed[word], axis = 0))[0,0])
    return sum(dists)

def findMostDivergent(embeddingsDict):
    words = []
    for embeds in embeddingsDict.values():
        for embed in embeds:
            words += list(embed.wv.vocab.keys())
    words = set(words)
    print("Found {} words to compare".format(len(words)))
    return sorted([(w, findDiverence(w, embeddingsDict)) for w in words], key = lambda x: x[1], reverse=True)

def getDivergenceDF(word, embeddingsDict):
    dists = []
    cats = sorted(set(embeddingsDict.keys()))
    dists = {}
    print(word)
    for cat in cats:
        dists[cat] = []
        for embed in embeddingsDict[cat][1:]:
            dists[cat].append(np.abs(1 - sklearn.metrics.pairwise.cosine_similarity(np.expand_dims(embeddingsDict[cat][0][word], axis = 0),
                                                                             np.expand_dims(embed[word], axis = 0))[0,0]))
    return pandas.DataFrame(dists, index = cats)


def compareModels(df, category, sort = True):
    """If you are using time as your category sorting is important"""
    embeddings_raw = {}
    cats = sorted(set(df[category]))
    for cat in cats:
        #This can take a while
        print("Embedding {}".format(cat), end = '\r')
        subsetDF = df[df[category] == cat]
        #You might want to change the W2V parameters
        embeddings_raw[cat] = gensim.models.word2vec.Word2Vec(subsetDF['no_lemma_normalized_sents'].sum())
    #These are much quicker
    embeddings_aligned = {}
    for catOuter in cats:
        embeddings_aligned[catOuter] = [embeddings_raw[catOuter]]
        for catInner in cats:
            embeddings_aligned[catOuter].append(smart_procrustes_align_gensim(embeddings_aligned[catOuter][-1], embeddings_raw[catInner]))
    return embeddings_raw, embeddings_aligned

# COLLOCATIONS

def get_text_collocation(df):
    return nltk.Text(df.no_lemma_normalized_tokens.sum())

def get_concordance(text, word):
    index = nltk.text.ConcordanceIndex(text)
    return index.print_concordance(word)

def get_context(text, words):
    index = nltk.text.ContextIndex(text)
    return index.common_contexts(words)

def get_count(text,word):
    return text.count(word)

def plot_count(full_df, years, word):
    # counts
    years = sorted(years)
    counts = []
    for y in years:
        tmp = full_df[full_df.year==y]
        text = get_text_collocation(tmp)
        c = get_count(text,word)
        counts.append(c)
    sns.lineplot(x=years, y=counts)
    plt.title('Frequency of {} in TAL'.format(word))
    plt.show()

def plot_two_count(full_df, years, word, word2):
    # counts
    years = sorted(years)
    counts = []
    counts2=[]
    for y in years:
        tmp = full_df[full_df.year==y]
        text = get_text_collocation(tmp)
        c = get_count(text,word)
        c2 = get_count(text,word2)
        counts2.append(c2)
        counts.append(c)
    sns.lineplot(x=years, y=counts, label = word)
    sns.lineplot(x=years, y=counts2, label=word2)
    plt.title('Frequency of "{}" and "{}" in TAL'.format(word,word2))
    plt.show()

def print_collocation(df, wordlist, concordance=False, context=True):
    text = get_text_collocation(df)
    
    for w in wordlist:
        print('Word: {}'.format(w))
        if concordance:
            print('Concordance: ')
            get_concordance(text, w)
            print()
        if context:
            print('Common context: ')
            get_context(text, [w])
        print()
        print()

def agg_contexts(dfs,years,wordlist):
    final = []
    for df in dfs:
        text = get_text_collocation(df)
        tmp = {}
        for w in wordlist:
            x = list(get_context(text, [w]).keys())
            tmp[w] = x
        final.append(tmp)
        
    new = pd.DataFrame(final)
    new['year'] = years
    return new

def count_contexts(dfs,years, wordlist):
    final = []
    for df in dfs:
        text = get_text_collocation(df)
        tmp = {}
        for w in wordlist:
            x = list(get_context(text, [w]).keys())
            tmp[w] = {}
            for i in x:
                first = i[0]
                second = i[1]
                tmp[w][first] = tmp[w].get(first,0) + 1
                tmp[w][second] = tmp[w].get(second,0) + 1
        final.append(tmp)
        
    new = pd.DataFrame(final)
    new['year'] = years
    return new
        
def plot_dispersion(df,wordlist):
    text = get_text_collocation(df)
    sns.reset_orig() #Seaborn messes with this plot, disabling it
    text.dispersion_plot(wordlist)
    sns.set() #Re-enabling seaborn

def tag_sents_pos(sentences):
    """
    function which replicates NLTK pos tagging on sentences.
    """
    nlp = spacy.load("en")
    new_sents = []
    for sentence in sentences:
        new_sent = ' '.join(sentence)
        new_sents.append(new_sent)
    final_string = ' '.join(new_sents)
    doc = nlp(final_string)
    
    pos_sents = []
    for sent in doc.sents:
        pos_sent = []
        for token in sent:
            pos_sent.append((token.text, token.tag_))
        pos_sents.append(pos_sent)
    
    return pos_sents

def most_common_adj(df, word):
    if 'POS_sents' not in df.columns:
        df['POS_sents'] = df.apply(lambda x: tag_sents_pos(x['tokenized_sents']), axis=1)
    NTarget = 'JJ'
    NResults = {}
    for entry in df['POS_sents']:
        for sentence in entry:
            for (ent1, kind1),(ent2,kind2) in zip(sentence[:-1], sentence[1:]):
                if (kind1,ent2.lower())==(NTarget,word):
                    if ent1 in NResults.keys():
                        NResults[ent1] +=1
                    else:
                        NResults[ent1] =1
                else:
                    continue
    return NResults

def make_verb_dict(verb, df):
    verb_dict = {}
    nlp = spacy.load("en")
    for index, row in df.iterrows():
        year = str(row['year'])
        if year not in verb_dict.keys():
            verb_dict[year] = ([],[])
        text = ' '.join(row['tokenized_text'])
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            subject = 0
            object_ = 0
            # if the verb or the root of the sentence is the word
            if chunk.root.head.text == verb:
                # we find the subjects and objects around the word,
                # and if it does exist, add it to the tuple
                if chunk.root.dep_ == 'nsubj':
                    subject = chunk.root.text
                if chunk.root.dep_ == 'dobj':
                    object_ = chunk.root.text
                if subject is not 0:
                    verb_dict[year][0].append(subject)
                if object_ is not 0:
                    verb_dict[year][1].append(object_)
    return verb_dict


## EMBEDDING BIAS

def mk_rep_group_vec(model, wordlist):
    vecs = []
    words = []
    for w in wordlist:
        try:
            v = model[w]
            vecs.append(v)
            words.append(w)
        except KeyError:
            continue
    #print('Caught words: {}'.format(words))
    return np.mean(vecs, axis=0)

def l2_norm_diff(model, rep_vec, wordlist):
    diffs = []
    words = []
    for w in wordlist:
        try:
            v = model[w]
            d = np.linalg.norm(rep_vec - v)
            diffs.append(d)
            words.append(w)
        except KeyError:
            continue
    #print('Caught words: {}'.format(words))
    return np.mean(diffs, axis=0)

def plot_bias(pre_dict, post_dict, title, wordlist_dict):
    diffs = []
    for k,v in wordlist_dict.items():
        tmp = {}
        tmp['category'] = k
        tmp['type'] = 'before event'
        tmp['bias'] = l2_norm_diff(pre_dict['model'], pre_dict['compare_vec'], v) - l2_norm_diff(pre_dict['model'], pre_dict['group_vec'], v)
        diffs.append(tmp)
        tmp = {}
        tmp['category'] = k
        tmp['type'] = 'after event'
        tmp['bias'] = l2_norm_diff(post_dict['model'], post_dict['compare_vec'], v) -l2_norm_diff(post_dict['model'], post_dict['group_vec'], v)
        diffs.append(tmp)
    df = pd.DataFrame(diffs)
    sns.catplot(x="category", y="bias", hue="type", data=df,
                height=6, kind="bar", palette="muted")
    plt.title(title)

# most similar words
def agg_sim_words(models,periods,wordlist):
    final = []
    for model in models:
        tmp = {}
        for w in wordlist:
            try:
                x = model.most_similar(w)
                tmp[w] = x
            except KeyError:
                tmp[w] = 'NA'
        final.append(tmp)
        
    new = pd.DataFrame(final)
    new['period'] = periods
    return new

## WORD NETWORK

def wordCooccurrence(sentences, makeMatrix = False):
    words = set()
    for sent in sentences:
        words |= set(sent)
    wordLst = list(words)
    wordIndices = {w: i for i, w in enumerate(wordLst)}
    wordCoCounts = {}
    #consider a sparse matrix if memory becomes an issue
    coOcMat = np.zeros((len(wordIndices), len(wordIndices)))
    for sent in sentences:
        for i, word1 in enumerate(sent):
            word1Index = wordIndices[word1]
            for word2 in sent[i + 1:]:
                coOcMat[word1Index][wordIndices[word2]] += 1
    if makeMatrix:
        return coOcMat, wordLst
    else:
        coOcMat = coOcMat.T + coOcMat
        g = nx.convert_matrix.from_numpy_matrix(coOcMat)
        g = nx.relabel_nodes(g, {i : w for i, w in enumerate(wordLst)})
        return g

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def graph_nx_word(df, colname, edge_weight, word):
    g = wordCooccurrence(df[colname].sum())
    g.remove_edges_from([(n1, n2) for n1, n2, d in 
                     g.edges(data = True) if d['weight'] <= edge_weight])

    g.remove_nodes_from(list(nx.isolates(g)))
    giant = max(connected_component_subgraphs(g), key=len) # keep just the giant connected component
    # get neighbors of word
    neighbors = giant.neighbors(word)
    g_word = giant.subgraph(neighbors)
    layout = nx.spring_layout(g_word, weight='weight', iterations= 100, k = .3)
    fig, ax = plt.subplots(figsize = (10,10))
    maxWeight = max((d['weight'] for n1, n2, d in g_word.edges(data = True)))
    minWeight = min((d['weight'] for n1, n2, d in g_word.edges(data = True)))
    nx.draw(g_word, ax = ax, pos = layout, labels = {n:n for n in g_word.nodes()},
            width = [(d['weight'] - minWeight + .7) / maxWeight for n1, n2, d in \
                    g_word.edges(data = True)], 
            alpha = 0.75, 
            font_size = 12,
            font_color = 'black',
            edge_color = 'black',
            cmap = plt.get_cmap('viridis'))
