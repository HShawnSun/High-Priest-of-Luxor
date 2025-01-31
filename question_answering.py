import pandas as pd
import numpy as np 
import sklearn
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import os

from util import text_preprocessing

# Use absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
small_talk_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-small_talk.csv')
qa_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-QA.csv')

# question_answering.py

import pandas as pd
import os
from util import text_preprocessing

def answer_Q(question, threshold=0.01):
    """
    Find answer for given question using similarity matching
    """
    try:
        # Read CSV with encoding that handles special characters
        df = pd.read_csv(qa_dataset, encoding='latin1').fillna("Unknown question")
        
        # Preprocess questions
        df['processed_Q'] = df['Question'].apply(lambda x: text_preprocessing(str(x)) if pd.notna(x) else "")
        processed_input = text_preprocessing(str(question) if pd.notna(question) else "")
        
        # Calculate similarities
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        question_vectors = vectorizer.fit_transform(df['processed_Q'])
        input_vector = vectorizer.transform([processed_input])
        
        similarities = cosine_similarity(input_vector, question_vectors)[0]
        max_sim = similarities.max()
        
        if max_sim >= threshold:
            return df.iloc[similarities.argmax()]['Answer']
        return 'NOT FOUND'
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return 'NOT FOUND'


