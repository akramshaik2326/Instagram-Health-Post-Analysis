#!/usr/bin/env python
# coding: utf-8

#importing pandas to read and manage csv file 
import pandas as pd
import numpy as np
# importing matplotlib to visulaize data
import matplotlib.pyplot as plt

#libraray for preprocessing of tweets
# importing stopwords, wornet, lemmatizer from nltk.corpus library
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

# importing re library to apply regular expressions, and string to use filter punctuations
import string
import re
from string import punctuation

#importing warnings library to ignore them keep file clean visually to understand
import warnings
warnings.filterwarnings("ignore")

#downloading different word data to preprocess
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
words = set(nltk.corpus.words.words())

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #print('stopfree: ',stop_free)
    hash_free = re.sub(r"#\S+", "", stop_free)
    mention_free = re.sub(r"@\S+", "", hash_free)
    num_free = re.sub(r"\d+","",mention_free)
    RT_free = re.sub("RT[\s]+", "", num_free)
    #print('number free: ', num_free)
    link_free = re.sub(r"https\S+", "", RT_free)
    punc_free = ''.join(ch for ch in link_free if ch not in exclude)
    otherlang_free = ''.join(ch for ch in punc_free if ch in words or ch == ' ')
    #print('punc: ',punc_free)
    #print('other: ',otherlang_free)
    normalized = " ".join(lemma.lemmatize(word) for word in otherlang_free.split())
    #print('normalized: ',normalized)
    len_free = ''.join(word for word in otherlang_free if len(word)>=1 )
    #print('len free:',len_free)
    len_free1 = len_free.split()
    #print(len_free1)
    lang_free = ' '.join(word for word in len_free1 if wordnet.synsets(word))
    #print(lang_free)
    y = lang_free.split()
    filtered=[]
    for i in y:
        if len(i)>2:
            filtered.append(i)
    return filtered