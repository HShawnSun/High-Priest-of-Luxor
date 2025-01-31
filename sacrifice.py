import pandas as pd
import numpy as np 
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
import os
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from datetime import timedelta

from util import text_preprocessing, correct
import re
from sklearn.linear_model import LogisticRegression

# Use absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')


def get_sacrifice(user_input):
    match = re.search(r'\d+', user_input)
    if match:
        start_index = match.start()
        end_index = re.search(r'\b(to|for|and|or|but|nor|so|yet)\b|[^\w\s]', user_input[start_index:])
        if end_index:
            sacrifice = user_input[start_index:start_index + end_index.start()]
        else:
            sacrifice = user_input[start_index:]
    else:
        sacrifice = ""
    
    return sacrifice
    
def count_sacrifice(user_input):
    numbers = re.findall(r'\d+', user_input)
    if len(numbers) == 0:
        return 0
    elif len(numbers) == 1:
        return 1
    else:
        return 2
