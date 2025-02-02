
import numpy as np 
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import pandas as pd
from nltk.tokenize import word_tokenize
import os

from util import text_preprocessing

# Use absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
small_talk_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-small_talk.csv')
qa_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-QA.csv')
name_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-name.csv')

CHNAGE_NAME = ["rename", "switch", "change", "call"]
NAME = ["call", "me", "change", "my", "name", "to", "please", "rename", "switch", "yes", "sure"]

def check_name_change(input):
    text_tokens = word_tokenize(input)
    if not set(text_tokens).isdisjoint(CHNAGE_NAME):
        return True
    else:
        return False

def name_change(input):
    text_tokens = word_tokenize(input)
    user_name = [i for i in text_tokens if not i.lower() in NAME and i.isalpha() and not i.lower() in stopwords.words('english')]
    user_name = (' ').join(user_name)
    return user_name

def name_response(query, threshold):

    df = pd.read_csv(name_dataset)

    # TF-IDF
    tfidf_vec = TfidfVectorizer(analyzer='word')
    X_tfidf = tfidf_vec.fit_transform(df['Question']).toarray()
    df_tfidf = pd.DataFrame(X_tfidf, columns = tfidf_vec.get_feature_names_out())

    # process query 
    input_tfidf = tfidf_vec.transform([query.lower()]).toarray()

    # cosine similarity
    cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric = 'cosine')
    
    if cos.max() >= threshold:
        return 'RESPOND'
    else:
        return 'NOT FOUND'
    
# if __name__ == "__main__":
    # print(talk_response("What is up", 0.1))
