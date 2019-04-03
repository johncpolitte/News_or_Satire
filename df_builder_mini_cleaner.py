import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import re
import itertools


def cnn_cleaner(cnn_df):
    '''
    Partially cleans the articles in the cnn_df by removing 'CNN'
    some punctuation characters and getting rid of capital letters.
    Also adds dummy variables for Satire, Fox and CNN columns.
    Lastly only takes first 5000 articles
    input: pandas DataFrame (dirty)
    output: pandas DataFrame (cleaned)
    '''
    # Converts article bodies to list
    cnn_list = list(cnn_df.content)
    # Creates empty list to append to
    clean_cnn_list = []
    # Iterate through list of articles for cleaning
    for article in cnn_list:
    # Removes CNN from articles
        clean_article = re.sub('CNN', '', article)
    # Removes punctuation characters
        clean_article2 = re.sub('[)(,.]', '', clean_article)
    # Removes capital letters and appends to clean_cnn_list
        clean_cnn_list.append(clean_article2.lower())
    # Converts list to DataFrame
    clean_cnn_df = pd.DataFrame(clean_cnn_list, columns=['Article'])
    # Creates dummy columns
    clean_cnn_df['Satire'] = 0
    clean_cnn_df['CNN'] = 1
    clean_cnn_df['Fox'] = 0
    return clean_cnn_df[0:5000]

def fox_cleaner(fox_df):
    '''
    Partially cleans the Fox DataFrame by getting rid of
    capital letters and removing some punctuation. Also Creates
    dummy variables for Satire, CNN, and Fox
    input: pandas DataFrame (dirty)
    output: pandas DataFrame (clean)
    '''
    # Converts content column to a list of the content in the articles
    fox_list = list(fox_df.content)
    # Creates empty list to append cleaned articles to
    clean_fox_list = []
    # Iterate through each article in the fox_list
    for article in fox_list:
    # Removes some of the punctuation
        clean = re.sub('[)(,.]', '', article)
    # Adds 
        clean_fox_list.append(clean.lower())
    clean_fox_df = pd.DataFrame(clean_fox_list, columns=['Article'])
    clean_fox_df['Satire'] = 0
    clean_fox_df['CNN'] = 0
    clean_fox_df['Fox'] = 1
    return clean_fox_df

def small_onion_clean(onion_df):
    onion_df = onion_df.drop('Title', axis=1)
    clean_onion_df = onion_df[0:5000]
    return clean_onion_df[['Article','Satire','CNN', 'Fox']]

def build_df(onion_df, fox_df, cnn_df):
    df_onion = small_onion_clean(onion_df)
    df_cnn = cnn_cleaner(cnn_df)
    df_fox = fox_cleaner(fox_df)
    df_final = pd.concat([df_onion, df_cnn, df_fox], axis = 0, ignore_index=True)
    return df_final
