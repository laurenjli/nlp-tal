# nlp-tal

We present a novel approach to understanding public sentiment regarding major events through analyzing a previously under-utilized data set: podcast transcripts.
By analyzing a popular American podcast, "This American Life," which mixes journalism and narratives regarding public and private events, we are able to garner American public opinion on important cultural events that are unseen in previous analysis of other corpora like newspaper articles and books.
At a high level, we find that the topics covered by "This American Life" remain stable over time.
However, when we examine language more closely around specific events, we can understand the way the changes in which Americans understand and internalize these topics.
In our analysis, we find and interpret shifts in language in "This American Life" transcripts before and after major events in the United States of America.

# Corpus

For sampling, we plan to acquire the text of all of the archived This American Life podcast [transcripts online](https://www.thisamericanlife.org/archive). An advantage is that this archive is a census of the This American Life episodes since 1995. Furthermore, This American Life is very unique in its inception and format in that it focuses on different themes each week and has different people around the country sharing stories so there is a diverse range of perspectives in the corpus. However, this corpus is certainly not of all identity-related bodies of text in the U.S. since that time. Alternatives would be looking at other types of news columns or podcasts that focus on personal experience.

# Preliminary Analysis

Our preliminary analysis examines changes in the corpus over 5-year windows (from 1995-2019). We utilize a number of methods to show that the corpus does not change significantly at a high level.

- summary_statistics.ipynb: summary statistics of our corpus
- clustering.ipynb: Clustering analyses 
- topic_modeling.ipynb: Topic Modeling to analyze changes in topics over time
- doc_vec_space.ipynb: Word2Vec and Doc2Vec models to measure changes in words and documents over time

# Events Analysis

While the "This American Life" corpus stays relatively consistent over time when examining prologues or topics over five-year windows, our results show that when digging deeper to specific years and periods around events, the way that these topics and words are used actually shift due to current events happening the real world. In our analysis, we examine changes before and after 9/11, the Financial Crisis, and the 2016 election.

- 9_11_analysis.ipynb: 9/11 attack analysis
- financial_crisis_analysis.ipynb: Financial Crisis analysis
- election_analysis.ipynb: 2016 election analysis

# Pipeline

In order to conduct our analysis, we created a number of functions employed across our jupyter notebooks. These can be found in the source folder.

- source/cluster_fns.py: Functions related to clustering and topic modeling
- source/descript_scrape.py: Functions to Scrape Descriptions/Prologues of "This American Life" Episodes
- helper_functions.py: Functions used in our event analyses
