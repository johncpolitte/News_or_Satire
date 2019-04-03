import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import re



client = MongoClient()
db = client.capstone
collection = db.onion1
docs = collection.find()


def onion_cleaner(mongo_cursor):
    '''
    input: mongo_cursor - cursor for the onion documents that
    need cleaning
    output: dataframe with articles cleaned and stripped of characters
    Also adds Satire, CNN, and Fox columns for testing purposes
    '''
    dict_list = []
    # for loop iterates through mongo cursor and removes '_id'dict
    # it also breaks apart the dictionary into keys and values and
    # appends to the dict_list
    for x in mongo_cursor:
        x.pop('_id')
        q = list(x.items())
        dict_list.append(q)
    # Converts the dict_list to an array
    art_arr = np.array(dict_list)
    # Gets the shape of the array so it can be reshaped
    art_shape = art_arr.shape
    # Reshapes array so it is 2D
    exp = art_arr.reshape(art_shape[0],art_shape[2])
    # Creates DF with URL and Article columns
    df = pd.DataFrame(exp, columns=['Title', 'Article'])
    # Converts Article column to list for text processing
    art = list(df['Article'])
    clean_list = []
    # Removes unwanted characters in article text and then appends
    # to clean_list
    for sample in art:
        sample1 = re.sub('<br/>', '', sample)
        sample2 = re.sub('</p>', '', sample1)
        sample3 = re.sub('—', ' ', sample2)
        sample4 = re.sub('<em>', '', sample3)
        sample5 = re.sub('</em>', '', sample4)
        sample6 = re.sub('\xa0', '', sample5)
        sample7 = re.sub('<p>', '', sample6)
        sample8 = re.sub('sic', '', sample7)
        sample9 = re.sub('[)(,.]', '', sample8)
        clean_list.append(sample9.lower())
    # Adds cleaned articles back to DF
    df['Article'] = clean_list
    # Creates dummies columns for future testing
    df['Satire'] = 1
    df['CNN'] = 0
    df['Fox']= 0
    return df




if __name__ == '__main__':

    df = onion_cleaner(docs)

    df.to_csv(path_or_buf = 'data/onion_csv.csv')
