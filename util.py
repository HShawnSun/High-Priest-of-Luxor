# util.py

import re
import collections
import nltk
import random
import logging
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

def words(text): 
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

def load_training_data():
    default_text = """the quick brown fox jumps over the lazy dog
                     programming python coding software development
                     common spelling mistakes corrections"""
    try:
        with open('./dataset/text_check.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            logger.info("Successfully loaded training data")
            return text
    except FileNotFoundError:
        logger.warning("Training file not found. Using default text.")
        return default_text

# Initialize model
NWORDS = train(words(load_training_data()))

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    return set(deletes + transposes)

def known(words): 
    return set(w for w in words if w in NWORDS)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def correct(word):
    try:
        word = word.lower()
        candidates = (known([word]) or 
                     known(edits1(word)) or 
                     known_edits2(word) or 
                     [word])
        return max(candidates, key=NWORDS.get)
    except Exception as e:
        logger.error(f"Error in word correction: {str(e)}")
        return word

def emotion():
    """Returns a random emotion emoji"""
    try:
        emotions = [
            "🏺", "🪔", "🪶", "🪨", "🪵",
            "🪴", "🪷", "🪸", "🪹", "🪺", "🪻", "🪼",
            "🏛️", "🏟️",  "🕌", "🛕","🕍", "🏜️", "🏝️"
        ]
        return random.choice(emotions)
    except Exception as e:
        logger.error(f"Error generating emotion: {str(e)}")
        return "😊"

def time_response(query_type):
    try:
        current = datetime.now()
        if query_type == 'time':
            time_str = current.strftime("%I:%M %p")
            print(f">> Spock: It's {time_str} {emotion()}")
        elif query_type == 'today':
            date_str = current.strftime("%A, %B %d, %Y")
            print(f">> Spock: Today is {date_str} {emotion()}")
    except Exception as e:
        logger.error(f"Error in time response: {str(e)}")
        print(">> Spock: Sorry, I couldn't get that information right now 😕")

def text_preprocessing(text, type='lemmatization'):
    """
    Preprocess text with option for stemming or lemmatization
    Args:
        text: input text string
        type: 'stemming' or 'lemmatization'
    """
    try:
        # Initialize tools
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming/lemmatization
        if type == 'stemming':
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        else:  # default to lemmatization
            processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(processed_tokens)
        
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        return text