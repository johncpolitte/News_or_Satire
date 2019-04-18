# The Onion or Real News
## Introduction
This project is an investigation into if an article can be distinguished between “The Onion” and real news based on the text. Fake news is a hot topic in the national conscience these days, and it is important for people to be able to distinguish the two. This project addressed a small portion of this problem by addressing “The Onion.” “The Onion” is a satirical media company that publishes articles in a voice that are trying to mimic the tone and structure of traditional news reporting in a satirical way. Throughout “The Onions” existence, many prominent people and countries have fallen for their articles and believed them to be true. In 2012, “The Onion” ran an article that stated Kim Jong Un was the sexiest man alive for that year, and China picked up the story and ran with it as fact. There are even sub-reddits dedicated to examples of times people fell for different articles. In order to help people not fall for articles from “The Onion,” I built a model that can distinguish between articles from "The Onion", and actual news sources. 

    
    
To accomplish this task, I will use natural language processing and different modeling techniques to predict where the article was published. I will be using articles from "The Onion", "CNN", and "Fox News". The first step will be to build a model that can differentiate between satire and actual news. Then I will see if my model can differentiate between "CNN", and "Fox News".

## Repository Information
A more in depth look into this project can be found in the Onion_or_News_notebook.ipynb found here:
https://github.com/johncpolitte/News_or_Satire/blob/master/Onion_or_News_notebook.ipynb

All of the code for this project can be found in the src file here: https://github.com/johncpolitte/News_or_Satire/tree/master/src  

The helper_functions.py file contains all of the functions used in the jupyter notebook mentioned above. 

web_scraper.py is the file used to scrape "The Onion" articles from their website and store them in a MongoDB. This was done using an EC2 on AWS. 

onion_to_csv.py converts the MongoDB into a csv file in the EC2 so it could be transferred to a local machine for further investigation. 

## Previous Work
Similar studies have been done to this one. One of them are described below: 

I found a similar study done by Dawn Graham. Her results can be found here: https://towardsdatascience.com/fake-news-or-not-edad1552aa02. She used natural language processing to build a logistic regression model that would predict whether or not an article came from r/TheOnion or r/nottheonion based on the headlines of the articles. 

My approach is different from the previous approach's because I am using the content of news articles from the respective article sources. I believe my approach will be more successful because the articles have more context then just a headline.


## Data Source
The CNN and Fox news articles are coming from a dataset that I found online. The data is stored as a CSV and can be found here: https://www.kaggle.com/lenamkhanh/analyzing-the-news/code. There are around 5,000 Fox News articles and around 15,000 CNN articles. 
    
For the Onion articles, I built a webscraper that scraped the content of each article and stored them into MongoDB. Ideally I will have upwards of 5,000 articles, so that I can have the same size set of articles for each "news" source. 


## Exploritory Data Analysis
I found that the average word count for each of the three sources of articles differed dramatically. Because of this, it was necessary to use a normalization process in the vectorization of the articles. The average word counts can be found in the bar chart below.

<img src="images/README_IMG/word_count.png"
    style="float: left; margin-right: 10px;" />

In the Logistic Regression Exploration section I ran two separate tests to confirm the normalization process was working. 

## Model Results

<img src="images/README_IMG/log_reg_con_mat.png"
    style="float: left; margin-right: 10px;" />

<img src="images/README_IMG/NB_con_mat.png"
    style="float: left; margin-right: 10px;" />

<img src="images/README_IMG/roc_curve.png"
    style="float: left; margin-right: 10px;" />

## Logistic Regression Exploration

<img src="images/README_IMG/beta_coef_election_stop_words.png"
    style="float: left; margin-right: 10px;" />

<img src="images/README_IMG/beta_cof_log_reg.png"
    style="float: left; margin-right: 10px;" />

To confirm that the data was being normalized correctly I performed two separate tests. The first test was to cut a CNN article to be the length of an average Onion article. Using an article that was 196 words, the logistic regression model still classified the article to be a news article with a probability of 80%. 

The second test was to find the average magnitude of all the Onion article vectors, and news article vectors after the articles were vectorized using the TFIDF Vectorizor. The average magnitude for the Onion article vectors and news article vectors can be found in the bar graph below. Both had an average magnitude of 1. With both of these tests I was able to confirm that the normalization process was working correctly. 

<img src="images/README_IMG/vector_mag.png"
    style="float: left; margin-right: 10px;" />









<img src="images/README_IMG/proba_word_removal.png"
    style="float: left; margin-right: 10px;" />



