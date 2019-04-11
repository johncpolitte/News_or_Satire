import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import re
import itertools

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


from scipy.sparse.linalg import norm, svds
from scipy.sparse import csr_matrix, find
from scipy.stats import beta


import matplotlib.pyplot as plt

def build_df(onion_df, fox_df, cnn_df):
    '''
    Runs all the cleaner functions on their specific dataframes and combines
    them into one DataFrame
    input: three dirty pandas DataFrames (onion, fox, and cnn)
    output: one big DataFrame
    '''
    # Performs the three preliminary cleaning functions and creates 1 data from
    # from the three independent data frames
    df_onion = small_onion_clean(onion_df)
    df_cnn = cnn_cleaner(cnn_df)
    df_fox = fox_cleaner(fox_df)
    df_final = pd.concat([df_onion, df_cnn, df_fox], axis = 0, ignore_index=True)
    # Returns df_final after dropping all of the articles that are less
    # than 50 words because some onion articles were not scrapped properly and
    # some of the fox articles had no content
    # Short_index returns a list of the indices for the short articles
    # so they are just dropped from the DataFrame
    return df_final.drop(short_index(find_short_articles(df_final)))


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
    '''
    Takes the first 5000 articles in the onion dataframe and creates a
    DataFrame with the appropriate columns
    input: pandas DataFrame
    output: pandas DataFrame (clean)
    '''
    onion_df = onion_df.drop('Title', axis=1)
    clean_onion_df = onion_df[0:5000]
    return clean_onion_df[['Article','Satire','CNN', 'Fox']]


def find_short_articles(df_final):
    '''
    Takes DataFrame where article content is in column
    titles 'Article'and then returns article len for each article.
    input: DataFrame with column 'Article'
    output: list of article lengths
    '''
    count = 0
    words = 0
    article_len=[]
    for x in list(df_final.Article):
        length = len(x.split())
        article_len.append(str(length))
    return article_len

def short_index(article_lengths):
    '''
    Takes in a list of article lengths and returns a list of indices where
    the article has less than 50 words
    input: list of article lengths
    output: list of indices corresponding to the short articles
    '''
    count = 0
    idx_list = []
    for idx, art in enumerate(article_lengths):
        if int(art) < 50:
            idx_list.append(idx)
            count+=1
    return idx_list



def final_cleaner(df):
    '''
    Removes more features like punctuation and words that
    should not be in there
    Input: df_final
    output: df_final with cleaned articles
    '''
    article_list = list(df.Article)
    clean_list = []
    for article in article_list:
        # regex not a-z or whitespace
        #samp1 = re.sub('[''\”\“\‘;:\'\'\'•·%$!&+}{|><_…/\’*0123456789\`]', '', article)
        #samp2 = re.sub('[-—]', ' ', samp1)
        samp1 = re.sub(r'\W+', ' ', article)
        samp2 = re.sub('[0123456789_]', '', samp1)
        samp3 = samp2.replace('news','')
        clean_list.append(samp3.replace('fox',''))
    df['Article'] = clean_list
    return df

def single_article_cleaner(single_article):
    '''
    Removes more features like punctuation and words that
    should not be in there
    Input: df_final
    output: df_final with cleaned articles
    '''
    clean_list = []
    samp = single_article.lower()
    samp1 = re.sub(r'\W+', ' ', samp)
    samp2 = re.sub('[-—]', ' ', samp1)
    samp3 = samp2.replace('news','')
    clean_list.append(samp3.replace('fox',''))
    return clean_list


def word_count(content_list):
    '''
    Takes in a list of articles that is a list of strings
    and then returns an average word count for each article.
    input: list of strings
    output: average word count per article
    '''
    count = 0
    words = 0
    for x in content_list:
        length = len(x.split())
        words += length
        count += 1
    return words/count



def word_count_graph(word_counts_df):
    '''
    Creates bar graph of the average number of words per article from each of
    the three sources
    '''
    fig, ax = plt.subplots()
    plt.bar(word_counts_df.Source.values,word_counts_df.Average_Word_Count, color ='rby')
    ax.set_title("Average Word Count Based on News Source", fontsize = 15)
    ax.set_ylabel("Average Word Count", fontsize = 15)
    ax.set_xlabel("Source", fontsize = 15)

def main_tokenize(doc):
    '''
    Tokenization function for the TFIDF vectorizor
    input: string of article
    output: list of article tokens that have been stemmed
    '''
    # Tokenizes each word in the document
    tokens = word_tokenize(doc)
    # Defines an empty list to append the stemmed words to
    cleaned_docs = []
    # Creates PorterStemmer object
    porter = PorterStemmer()
    # For loop to iterate through tokens in document
    for word in tokens:
    # Removes tokens that are just one character
        if len(word) < 2:
            tokens.remove(word)
    # Removes stop words from articles that are in test_stop_words
    # list and appends porter stemmed words to the cleaned doc list
        else:
            if word not in test_stop_words:
                stem_word = porter.stem(word)
                if stem_word not in test_stop_words:
                    cleaned_docs.append(stem_word)
    # Returns list of tokens
    return cleaned_docs

def tokenize_no_stemmer(doc):
    '''
    Tokenization function for the TFIDF vectorizor. This version of the tokenizer has no
    stemming function
    input: string of article
    output: list of article tokens that have been stemmed
    '''
    # Tokenizes each word in the document
    tokens = word_tokenize(doc)
    # Defines an empty list to append the stemmed words to
    cleaned_docs = []
    # Creates PorterStemmer object
    porter = PorterStemmer()
    # For loop to iterate through tokens in document
    for word in tokens:
    # Removes tokens that are just one character
        if len(word) < 2:
            tokens.remove(word)
    # Removes stop words from articles that are in test_stop_words
    # list and appends porter stemmed words to the cleaned doc list
        else:
            if word not in test_stop_words:
                    cleaned_docs.append(word)
    # Returns list of tokens
    return cleaned_docs

def tokenize_election(doc):
    '''
    Tokenization function for the TFIDF vectorizor
    input: string of article
    output: list of article tokens that have been stemmed
    '''
    # Tokenizes each word in the document
    tokens = word_tokenize(doc)
    # Defines an empty list to append the stemmed words to
    cleaned_docs = []
    # Creates PorterStemmer object
    porter = PorterStemmer()
    # For loop to iterate through tokens in document
    for word in tokens:
    # Removes tokens that are just one character
        if len(word) < 2:
            tokens.remove(word)
    # Removes stop words from articles that are in election_stop_words
    # list and appends porter stemmed words to the cleaned doc list
        else:
            if word not in election_stop_words:
                stem_word = porter.stem(word)
                if stem_word not in election_stop_words:
                    cleaned_docs.append(stem_word)
    # Returns list of tokens
    return cleaned_docs


def model_performance_score(model, testing_data, testing_labels):
    y_pred = model.predict(testing_data)
    con_mat = confusion_matrix(testing_labels, y_pred)
    recall = con_mat[1,1]/(con_mat[1,1]+con_mat[1,0])
    precision = con_mat[1,1]/(con_mat[1,1]+con_mat[0,1])
    F1 = ((precision*recall)/(precision+recall))*2
    return 'F1 = {}, Precision = {}, Recall = {}'.format(F1, precision, recall)



def high_low_beta_coef(vect_object, model_object):
    '''
    This function takes in a document frequency vector and a trained model object and generates a list
    of the words with the 10 highest and lowest beta coef. As an intermediary step a dataframe with
    all of the words in the vector and their corresponding beta coefficients.
    input:
            vect_object = inverse document frequency vector ('vect')
            model_object = trained logistic regression model object ('log_reg')
    output:
            top_bot_words = list of the words that correspond to the 10 highest and 10 lowest beta coefficients
            bot_top_coef = list of the 10 lowest and highest beta coefficients
            sort = dataframe of the words in the vector space and their beta coefficients
    '''
    # Creates a data frame for each word and its corresponding
    feature_df = pd.DataFrame(vect_object.get_feature_names())
    feature_df['Beta_Coef'] = model_object.coef_.reshape(model_object.coef_.shape[1],)
    feature_df['Word'] = feature_df[0]
    beta_feature_df = feature_df[['Beta_Coef', 'Word']]
    sort = beta_feature_df.sort_values('Beta_Coef')
    bot_top_coef = list(sort['Beta_Coef'][0:10]) + list(sort['Beta_Coef'][-10:])
    top_bot_words = list(sort['Word'][0:10]) + list(sort['Word'][-10:])
    return top_bot_words, bot_top_coef, sort

def avg_tfidf_mag(training_data, training_labels):
    '''
    Calculates the average magnitude of the TFIDF vector for Satire and News
    categories
    input: sparse matrix for training
    '''
    row_norms=[]
    for row in training_data:
        row_norms.append(norm(row))
    sums_arr =np.array(row_norms)
    ytr_arr = np.array(training_labels)
    tot_sat_art = np.sum(ytr_arr)
    tot_news_art = len(ytr_arr) - tot_sat_art
    Sat_avg_mag = sums_arr[ytr_arr==1].sum()/tot_sat_art
    News_avg_mag = sums_arr[ytr_arr==0].sum()/tot_news_art
    return Sat_avg_mag, News_avg_mag

def onion_prob_word_removal(vect_object, clean_sample):
    vect_sample = vect_object.transform(clean_sample)
    row_idx, col_idx, val = find(vect_sample)
    row_col_list = list(zip(list(row_idx),(col_idx)))
    feat_array = np.array(vect_object.get_feature_names())
    probas_word_removal = []
    for idx in row_col_list:
        vect_sample2 = vect_object.transform(clean_sample)
        vect_sample2[idx] = 0
        prb = log_reg.predict_proba(vect_sample2)
        sat_prob = prb[0][1]
        probas_word_removal.append(sat_prob)
    countmin = 0
    countmax = 0
    for x in probas_word_removal:
        if x == min(probas_word_removal):
            minword = countmin
        if x == max(probas_word_removal):
            maxword = countmax
        countmin += 1
        countmax += 1
    return probas_word_removal,feat_array[col_idx[minword]], feat_array[col_idx[maxword]]


election_stop_words = ['serious','any', 'ours', 'go', 'do', 'else', 'while', 'somehow', 'seem', 'front', 'thick', 'once', 'system',
 'latter', 'amongst', 'hence', 'un', 'cannot', 'more', 'eight', 'he', 'seems', 'it', 'hereafter', 'last', 'here',
 'beyond', 'because', 'few', 'fill', 'his', 'further', 'sincere', 'their', 'made', 'fifty', 'whatever', 'whenever', 'been', 'describe', 'otherwise',
 'or', 'our', 'move', 'eg', 'over', 'per', 'amoungst', 'perhaps', 'you', 'beside', 'hundred', 'across',
 'which', 'where', 'anyone', 'anywhere', 'name', 'several', 'a', 'no', 'whence', 'mostly', 'so', 'call',
 'seemed', 'everyone', 'these', 'besides', 'whom', 'whereby', 'eleven', 'thereupon', 'twelve', 'when', 'former', 'most',
 'therein', 'had', 'hasnt', 'yourself', 'next', 'being', 'wherein', 'only', 'them', 'third', 'mine', 'show',
 'nobody', 'sometimes', 'somewhere', 'still', 'were', 'with', 'became', 'how', 'yourselves', 'her', 'much',
 'ltd', 'as', 'those', 'done', 'twenty', 'along', 'get', 'herself', 'interest', 'nor', 'however', 'same', 'side', 'whole', 'namely', 'might',
 'if', 'has', 'up', 'both', 'not', 'bottom', 'ourselves', 'via', 'whither', 'fifteen', 'your', 'mill',
 'someone', 'even', 'please', 'thus', 'under', 'are', 'in', 'etc', 'anyhow', 'after', 'hereupon', 'my', 'from',
 'through', 'before', 'own', 'against', 'below', 'throughout', 'although', 'herein', 'himself', 'noone', 'will',
 'also', 'thru', 'out', 'keep', 'something', 'there', 'nevertheless', 'nine', 'always', 'except', 'almost', 'some',
 'couldnt', 'hereby', 'indeed', 'detail', 'moreover', 'hers', 're', 'all', 'six', 'themselves', 'two', 'already', 'forty',
 'thereby', 'become', 'each', 'thence', 'within', 'nowhere', 'by', 'due', 'full', 'thin', 'us', 'anyway', 'other', 'among', 'this',
 'though', 'without', 'then', 'five', 'another', 'first', 'myself', 'every', 'at', 'de', 'toward', 'whereafter',
 'alone', 'beforehand', 'amount', 'ie', 'meanwhile', 'behind', 'must', 'now', 'others', 'many', 'be', 'con', 'an', 'formerly',
 'everywhere', 'therefore', 'find', 'to', 'together', 'could', 'elsewhere', 'about', 'three', 'am', 'since', 'me',
 'whose', 'ever', 'cry', 'becoming', 'whereas', 'see', 'well', 'back', 'everything', 'nothing', 'whether', 'itself', 'whoever', 'often', 'never', 'down', 'top', 'least', 'too',
 'of', 'during', 'inc', 'less', 'that', 'she', 'give', 'than', 'latterly', 'they', 'fire', 'found', 'the',
 'bill', 'thereafter', 'enough', 'very', 'have', 'its', 'who', 'anything', 'afterwards', 'around', 'upon',
 'but', 'either', 'again', 'should', 'what', 'into', 'none', 'would', 'can', 'for', 'put', 'empty', 'why', 'is', 'him', 'above', 'between', 'four', 'off',
 'cant', 'may', 'sometime', 'until', 'and', 'part', 'yet', 'onto', 'towards', 'neither', 'yours',
 'we', 'take', 'rather', 'on', 'such', 'was', 'ten', 'becomes', 'co', 'one', 'i', 'seeming', 'wherever', 'whereupon', 'sixty', 'trump', 'donald', 'hilary', 'clinton',
    'election', 'primary', 'mike', 'pence', 'vote', 'voter', 'poll', 'tim', 'kaine', 'republican', 'democrat', 'washington',
    'president', 'obama']

test_stop_words = ['serious','any', 'ours', 'go', 'do', 'else', 'while', 'somehow', 'seem', 'front', 'thick', 'once', 'system',
 'latter', 'amongst', 'hence', 'un', 'cannot', 'more', 'eight', 'he', 'seems', 'it', 'hereafter', 'last', 'here',
 'beyond', 'because', 'few', 'fill', 'his', 'further', 'sincere', 'their', 'made', 'fifty', 'whatever', 'whenever', 'been', 'describe', 'otherwise',
 'or', 'our', 'move', 'eg', 'over', 'per', 'amoungst', 'perhaps', 'you', 'beside', 'hundred', 'across',
 'which', 'where', 'anyone', 'anywhere', 'name', 'several', 'a', 'no', 'whence', 'mostly', 'so', 'call',
 'seemed', 'everyone', 'these', 'besides', 'whom', 'whereby', 'eleven', 'thereupon', 'twelve', 'when', 'former', 'most',
 'therein', 'had', 'hasnt', 'yourself', 'next', 'being', 'wherein', 'only', 'them', 'third', 'mine', 'show',
 'nobody', 'sometimes', 'somewhere', 'still', 'were', 'with', 'became', 'how', 'yourselves', 'her', 'much',
 'ltd', 'as', 'those', 'done', 'twenty', 'along', 'get', 'herself', 'interest', 'nor', 'however', 'same', 'side', 'whole', 'namely', 'might',
 'if', 'has', 'up', 'both', 'not', 'bottom', 'ourselves', 'via', 'whither', 'fifteen', 'your', 'mill',
 'someone', 'even', 'please', 'thus', 'under', 'are', 'in', 'etc', 'anyhow', 'after', 'hereupon', 'my', 'from',
 'through', 'before', 'own', 'against', 'below', 'throughout', 'although', 'herein', 'himself', 'noone', 'will',
 'also', 'thru', 'out', 'keep', 'something', 'there', 'nevertheless', 'nine', 'always', 'except', 'almost', 'some',
 'couldnt', 'hereby', 'indeed', 'detail', 'moreover', 'hers', 're', 'all', 'six', 'themselves', 'two', 'already', 'forty',
 'thereby', 'become', 'each', 'thence', 'within', 'nowhere', 'by', 'due', 'full', 'thin', 'us', 'anyway', 'other', 'among', 'this',
 'though', 'without', 'then', 'five', 'another', 'first', 'myself', 'every', 'at', 'de', 'toward', 'whereafter',
 'alone', 'beforehand', 'amount', 'ie', 'meanwhile', 'behind', 'must', 'now', 'others', 'many', 'be', 'con', 'an', 'formerly',
 'everywhere', 'therefore', 'find', 'to', 'together', 'could', 'elsewhere', 'about', 'three', 'am', 'since', 'me',
 'whose', 'ever', 'cry', 'becoming', 'whereas', 'see', 'well', 'back', 'everything', 'nothing', 'whether', 'itself', 'whoever', 'often', 'never', 'down', 'top', 'least', 'too',
 'of', 'during', 'inc', 'less', 'that', 'she', 'give', 'than', 'latterly', 'they', 'fire', 'found', 'the',
 'bill', 'thereafter', 'enough', 'very', 'have', 'its', 'who', 'anything', 'afterwards', 'around', 'upon',
 'but', 'either', 'again', 'should', 'what', 'into', 'none', 'would', 'can', 'for', 'put', 'empty', 'why', 'is', 'him', 'above', 'between', 'four', 'off',
 'cant', 'may', 'sometime', 'until', 'and', 'part', 'yet', 'onto', 'towards', 'neither', 'yours',
 'we', 'take', 'rather', 'on', 'such', 'was', 'ten', 'becomes', 'co', 'one', 'i', 'seeming', 'wherever', 'whereupon', 'sixty']
