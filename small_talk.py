import pandas as pd
import numpy as np 
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import os

from util import text_preprocessing

# Use absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
small_talk_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-small_talk.csv')
qa_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-QA.csv')

def talk_response(query, threshold):

    df = pd.read_csv(small_talk_dataset)

    # TF-IDF
    tfidf_vec = TfidfVectorizer(analyzer='word')
    X_tfidf = tfidf_vec.fit_transform(df['Question']).toarray()
    df_tfidf = pd.DataFrame(X_tfidf, columns = tfidf_vec.get_feature_names_out())

    # process query 
    input_tfidf = tfidf_vec.transform([query.lower()]).toarray()

    # cosine similarity
    cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric = 'cosine')
    
    if cos.max() >= threshold:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0]) 
        return df['Answer'].loc[id]
    else:
        return 'NOT FOUND'
    
if __name__ == "__main__":
    print(talk_response("What is up", 0.1))

