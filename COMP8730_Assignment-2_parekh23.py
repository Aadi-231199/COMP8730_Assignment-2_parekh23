# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 21:01:03 2022

COMP8730_Assignment-2

@author: Aaditya Pradipbhai Parekh
Uwin ID:- parekh23

"""

import nltk
import numpy as np
import pandas as pd
import datetime
from nltk.corpus import brown
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
nltk.download('brown') 
nltk.download('punkt') 

SPELLING_ERROR_CORPUS_FILE = 'APPLING1DAT.643'

# Reading from news genre from Brown's Corpus
news_corp_sentence = list(brown.sents(categories='news'))
news_corp_sentence = [' '.join(sent) for sent in news_corp_sentence]
news_corp = ' '.join(news_corp_sentence)

# Reading the Birkbeck spelling error corpus
with open(SPELLING_ERROR_CORPUS_FILE, 'r') as file:
    l = file.read().splitlines()

l = [word.lower() for word in l]

spell_err_corp = [(line.split()[0], line.split()[1],
                          ' '.join(line.split()[2:]))
                         for line in l if line[0] != '$']
spell_err_corp = [[term[0], term[1], term[-1].replace('*', '').split()]
                         for term in spell_err_corp]

news_corp_tokenized = [sent_tokenize(str.lower(sent))[0] for sent in news_corp_sentence]
news_corp_tokenized = [word_tokenize(sent) for sent in news_corp_tokenized]

news_corp = [list(map(str.lower, word_tokenize(sent))) 
                 for sent in sent_tokenize(news_corp)]




def calculate_word_probability_phrase(ngram, spell_err_corp):
    ngram_prob = list()
    
    p_list = [(term[1], term[-1]) for term in spell_err_corp]
    w_list = [term[1] for term in spell_err_corp]
    
    for p_group in p_list:
        for w in w_list:
            expected_word = p_group[0]
            phrase = p_group[1]
            probability = ngram.score(w, phrase)
            ngram_prob_tuple = (w, expected_word, ' '.join(phrase), probability)
            ngram_prob.append(ngram_prob_tuple)
    
    return ngram_prob

# Building n-gram models where the value of n is {1, 2, 3, 5, 10}
n_values = [1, 2, 3, 5, 10]
for i in n_values:
    train_news_corpus, padded_news_sents = padded_everygram_pipeline(i, news_corp_tokenized)
    ngram = MLE(i)
    ngram.fit(train_news_corpus, padded_news_sents)
    ngram_prob = calculate_word_probability_phrase(ngram, spell_err_corp)
    
    ngram_probability_dataframe = pd.DataFrame(ngram_prob, columns = ['Words',
                                                          'Expected Words','Phrase','Probabilities'
                                                          ])
    
    top_ten_grouped = ngram_probability_dataframe\
        .groupby(['Phrase', 'Expected Words'])\
        .apply(lambda x: x.nlargest(10, ['Probabilities']))\
        .reset_index(drop=True)
    
    top_ten_grouped['success'] = np.where(
        top_ten_grouped['Words'] == top_ten_grouped['Expected Words'], 1, 0)
    
    top_ten_grouped['success_at_k'] = top_ten_grouped.groupby(
        ['Phrase', 'Expected Words'])['success'].transform(pd.Series.cumsum)
    
    s_at_k = top_ten_grouped.groupby(['Phrase', 'Expected Words'])\
        .nth([0,4,9])['success_at_k']
    
    average_S_at_K = s_at_k.groupby(['Phrase', 'Expected Words']).transform(np.mean)
    
    average_S_at_K = average_S_at_K.reset_index().drop_duplicates(ignore_index=True)
    
    average_S_at_K.to_csv('average_s@k.csv', index=False)
    
    s_at_one = top_ten_grouped.groupby(['Phrase', 'Expected Words'])\
        .nth(0)['success_at_k']
    average_s_at_one = np.mean(s_at_one)
    
    s_at_five = top_ten_grouped.groupby(['Phrase', 'Expected Words'])\
        .nth(4)['success_at_k']
    average_s_at_five = np.mean(s_at_five)
    
    s_at_ten = top_ten_grouped.groupby(['Phrase', 'Expected Words'])\
        .nth(9)['success_at_k']
    average_s_at_ten = np.mean(s_at_ten)
    
    average_result = pd.DataFrame(
        [(average_s_at_one, average_s_at_five, average_s_at_ten)],
        columns=['Average Success at 1', 'Average Success at 5',
                 'Average Success at 10'])
    
    print("Given n = " + str(i))
    print(average_result)
    
    start_time = datetime.datetime.now()
    end_time = datetime.datetime.now()
    print("Total Time : " + str(end_time - start_time))
