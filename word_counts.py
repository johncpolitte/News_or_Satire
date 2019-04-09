import matplotlib.pyplot as plt

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
