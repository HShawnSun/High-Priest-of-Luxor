import os
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from datetime import timedelta
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingRegressor

from question_answering import answer_Q
from name_management import name_change, check_name_change, name_response
from small_talk import talk_response
from util import correct, emotion, time_response
from deity import get_deity
from sacrifice import get_sacrifice, count_sacrifice
from sklearn.linear_model import LogisticRegression

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')
small_talk_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-small_talk.csv')
qa_dataset = os.path.join(dataset_dir, 'COMP3074-CW1-Dataset-QA.csv')

# Load small talk dataset
small_talk_df = pd.read_csv(small_talk_dataset)
small_talk_intents = small_talk_df.iloc[:, 0].tolist()

# Training data for intent classification
X_train = [
    # Small talk, labels: 0
    "hello how are you",
    "what's up",
    "nice to meet you",
    "my name is John",
    "you can call me Sarah",
    "please call me Mike",
    "hi there",
    "good morning",
    "i am feeling great",
    "good evening",
    "pleased to meet you",
    "what can you do",

    # Name management, labels: 1
    "change my name to Alex",
    "what is my name",
    "do you remember my name",
    "do you know my name",
    "who am I",
    "tell me who am i",
    "i guess you forget me",
    "call my name",
    "what's my name",
    
    # Questions, labels: 2
    "what county is Farmington Hills, MI in",  
    "what does a groundhog look for on groundhog day",  
    "what committees are joint committees",  
    "how many stripes on the flag",  
    "how many states and territories are within India",  
    "how is whooping cough distinguished from similar diseases",  
    "what county is Galveston in Texas",  
    "what cities are in the Bahamas",  
    "how many schools are in the Big Ten",  
    "how is single malt scotch made",  
    "what composer used sound mass",  
    "how do you play spades",  
    "what continent is Australia",  
    "what causes heart disease",  
    "what countries did immigrants come from during the immigration",  
    "how many people live in Atlanta, Georgia",  
    "what does it mean to be a commonwealth state",  
    "how long can you be in the supreme court",  
    "what county in Texas is Conroe located in",  
    
    # Time, labels: 3
    "what time is it",
    "what is the date today",
    "What's the current time?",
    "What's today's date?",
    "Do you know what time it is?",
    "What's the exact time at the moment?",
    
    # Sacrifice, labels: 4
    "I wish to offer a tribute to a deity.",
    "I have come to make an offering to the divine.",
    "I seek to present a gift to the gods.",
    "I bring this offering to honor a god.",
    "I desire to pay my respects to the gods with this sacrifice.",
    "I have come to dedicate this to a deity.",
    "I wish to venerate a god with my offering.",
    "I am here to present this as a token of devotion.",
    "I come bearing a sacrifice in service to the divine.",
    "I wish to consecrate this offering to a deity.",
    "I have brought this gift to honor the sacred.",
    "I seek to express my devotion through this act of sacrifice.",
    "I wish to worship a god",
    "I wish to serve a god",
    
    # Yes
    "yes",
    "sure",
    "of course",
    "absolutely",
    "indeed",
    "definitely",
    "yeah",
    "yep",
    "okay",
    "fine",
    "right",
    "affirmative",
    
    # No
    "no",
    "nope",
    "nah",
    "negative",
    "not really",
    "not sure",
    "not exactly",
    "not at all",
    "not necessarily",
    "not really",
    "not particularly",
    "not likely",
    
    # check sacrifice
    "Have I presented any gifts to the gods?",
    "Have I brought any tributes to the deities?",
    "Have I made any offerings in honor of the divine?",
    "Have I given anything in sacrifice to the gods?",
    "Have I shown any devotion through offerings?",
    "What have I provided to appease the deities?",
    "What sacrifices have I made to the divine?",
    "Have I contributed anything to the gods' favor?",
    "What have I presented as an offering to the deities?",
    "Have I honored the gods with any sacrifices?",
    
    # chisel off the sacrifice
    "I wish to remove my name and sacrifice from the wall.",
    "erase my name and offering from the wall.",
    "remove my name and tribute from the wall.",
    "chisel off my name and gift from the wall.",
    "sand off my name and devotion from the wall.",
    "clean off the dedication I made to the gods.",
    "scrub away the offering I presented to the divine.",
    "erase the tribute I gave to the deities.",
    "remove the gift I offered to the gods.",
    "scrape off the name and sacrifice I made.",
    
    # quit
    "bye",
    "goodbye",
    "farewell",
    "see you later",
    "I have to go",
    "I must leave",
    "I'm leaving",
    "I'm going now",
    "quit",
    "exit",
]

# Labels: 0 for small talk, 1 for name management, 2 for questions
y_train = [0]*12 + [1]*9 + [2]*19 + [3]*6 + [4]*14 + [5]*12 + [6]*12 + [7]*10 + [8]*10 + [9]*10


# Initialize vectorizer and classifier
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
classifier = LogisticRegression()
classifier.fit(X_train_vectors, y_train)

def get_intent(user_input):
    user_input = [correct(i) for i in user_input.split(' ')]
    user_input = (' ').join(user_input)
    user_input_vector = vectorizer.transform([user_input])
    intent = classifier.predict(user_input_vector)[0]
    if intent == 0:
        return 'small_talk'
    elif intent == 1:
        return 'name_management'
    elif intent == 2:
        return 'question'
    elif intent == 3:
        return 'time'
    elif intent == 4:
        return 'sacrifice'
    elif intent == 5:
        return 'yes'
    elif intent == 6:
        return 'no'
    elif intent == 7:
        return 'check_sacrifice'
    elif intent == 8:
        return 'chisel_off_the_sacrifice'
    elif intent == 9:
        return 'bye'
    else:
        return 'unknown'

if __name__ == "__main__":
    user_name = '(Unknown)'
    flag = True
    print(">> High Priest: I am the High Priest of Thebes. Welcome to the Royal Temple of Luxor!")
    print("           May I have your name? %s" % emotion())
    print('>> %s: ' % user_name, end="")
    user_input = input()
    intent = get_intent(user_input)
    if intent == 'bye':
        flag = False
    else:
        user_name = name_change(user_input)
        print(">> High Priest: Greetings, %s. I am at your service. %s" % (user_name, emotion()))

    while flag:
        print('>> %s: ' % user_name, end="")
        user_input = input()
        intent = get_intent(user_input)
        if intent != 'bye':
            if intent == 'name_management':
                if check_name_change(user_input):
                    user_name = name_change(user_input)
                    print(">> High Priest: I will now call you %s. %s" % (user_name, emotion()))
                else:
                    response = name_response(user_input, threshold=0.01)
                    if 'what' in user_input.lower() and 'name' in user_input.lower():
                        print(f">> High Priest: I remember your name. It is {user_name}. {emotion()}")
                    elif response != 'NOT FOUND':
                        print(">> High Priest: " + response + ' ' + emotion())
                    else:
                        print(">> High Priest: The issue is beyond the comprehension of mere mortals.  %s" % emotion())
                continue
            elif intent == 'time':
                if 'time' in user_input.lower():
                    time_response('time')
                elif 'today' in user_input.lower():
                    time_response('today')
                continue
            elif intent == 'question':
                print(">> High Priest: Let me consult the ancient scrolls for you. %s" % emotion())
                response = answer_Q(user_input, threshold=0.01)
                if response != 'NOT FOUND':
                    print(">> High Priest: " + response + ' ' + emotion())
                else:
                    response = talk_response(user_input, threshold=0.6)
                    if response != 'NOT FOUND':
                        print(">> High Priest: " + response + ' ' + emotion())
                    else:
                        print(">> High Priest: The issue is beyond the comprehension of mere mortals.  %s" % emotion())
                continue
            elif intent == 'sacrifice':
                print(">> High Priest: Do you know your deity's name? If not, tell me what you wish for, and I can help you find the deity that best matches your needs. %s" % emotion())
                print('>> %s: ' % user_name, end="")
                user_input = input()
                deity = get_deity(user_input)
                while True:
                    print(">> High Priest: What do you have to offer to the mighty one " + deity + ", who holds the power to bless you in this realm?" + " %s" % emotion())
                    print('>> %s: ' % user_name, end="")
                    user_input = input()
                    sacrifice = get_sacrifice(user_input)
                    sacrifice_number = count_sacrifice(user_input)
                    
                    if sacrifice_number == 0:
                        print(">> High Priest: Please specify the amount of your sacrifice using arabic numbers. %s" % emotion())
                        continue
                    elif sacrifice_number > 1:
                        print(">> High Priest: Please offer one sacrifice at a time. %s" % emotion())
                        continue
                    elif sacrifice_number == 1 and len(sacrifice) == 0:
                        print(">> High Priest: Please specify the item of your sacrifice. %s" % emotion())
                        continue
                    else:
                        print(">> High Priest: Are you sure to sacrifice " + sacrifice + " to " + deity +"?" + " %s" % emotion())
                        print('>> %s: ' % user_name, end="")
                        user_input = input()
                        user_input = [correct(i) for i in user_input.split(' ')]
                        user_input = (' ').join(user_input)
                        user_input_vector = vectorizer.transform([user_input])
                        intent = classifier.predict(user_input_vector)[0]
                        if intent == 5:
                            print(">> High Priest: Your sacrifice has been accepted. Do you wish to carve your name and sacrifice on the wall of the serene one " + deity +"'s altar room?" + " %s" % emotion())
                            print('>> %s: ' % user_name, end="")
                            user_input = input()
                            user_input = [correct(i) for i in user_input.split(' ')]
                            user_input = (' ').join(user_input)
                            user_input_vector = vectorizer.transform([user_input])
                            intent = classifier.predict(user_input_vector)[0]
                            if intent == 5:
                                print(f">> High Priest: Your name and sacrifice have been carved on the wall of the great spirit {deity}'s altar room. Feel free to check it out anytime! {emotion()}")
                            
                                # Define the file path for the sacrifices record
                                file_path = os.path.join(dataset_dir, 'sacrifices.json')
                            
                                # Ensure the file exists and initialize it if necessary
                                if not os.path.exists(file_path):
                                    with open(file_path, 'w') as f:
                                        json.dump([], f)
                            
                                # Load existing sacrifices or initialize an empty list
                                with open(file_path, 'r') as f:
                                    try:
                                        sacrifices = json.load(f)
                                    except json.JSONDecodeError:
                                        sacrifices = []
                            
                                # Append the new sacrifice record
                                sacrifices.append({"name": user_name, "sacrifice": sacrifice, "deity": deity})
                            
                                # Save the updated sacrifices list back to the file
                                with open(file_path, 'w') as f:
                                    json.dump(sacrifices, f, indent=4)
                                break
                            else:
                                print(f">> High Priest: Sorry that you refused this honor. {emotion()}")
                                break
                continue
            elif intent == 'check_sacrifice':
                with open(os.path.join(dataset_dir, 'sacrifices.json'), 'r') as f:
                    sacrifices = json.load(f)
                user_sacrifices = [sacrifice for sacrifice in sacrifices if sacrifice["name"] == user_name]
                if user_sacrifices:
                    last_sacrifice = user_sacrifices[-1]
                    print(f">> High Priest: Thou hast offered {last_sacrifice['sacrifice']} unto the deity {last_sacrifice['deity']}. {emotion()}")
                else:
                    print(">> High Priest: No record of thy sacrifice is found. %s" % emotion())
                continue
            elif intent == 'chisel_off_the_sacrifice':
                print(">> High Priest: Do you wish to chisel off your name and sacrifice from the wall? %s" % emotion())
                print('>> %s: ' % user_name, end="")
                user_input = input()
                user_input = [correct(i) for i in user_input.split(' ')]
                user_input = (' ').join(user_input)
                user_input_vector = vectorizer.transform([user_input])
                intent = classifier.predict(user_input_vector)[0]
                if intent == 5:
                    with open(os.path.join(dataset_dir, 'sacrifices.json'), 'r+') as f:
                        try:
                            sacrifices = [json.loads(line) for line in f]
                        except json.JSONDecodeError:
                            sacrifices = []
                        sacrifices = [sacrifice for sacrifice in sacrifices if user_name not in sacrifice]
                        f.seek(0)
                        f.truncate()
                        for sacrifice_record in sacrifices:
                            json.dump(sacrifice_record, f)
                            f.write('\n')
                    print(">> High Priest: Your name and sacrifice have been chiselled off the wall. %s" % emotion())
                else:
                    print(">> High Priest: Sorry that you refused this honour. %s" % emotion())
            else:  # intent == 'small_talk' or 'unknown'
                response = talk_response(user_input, threshold=0.8)
                if response != 'NOT FOUND':
                    print(">> High Priest: " + response + ' ' + emotion())
                else:
                    print(">> High Priest: The issue is beyond the comprehension of mere mortals.  %s" % emotion())
            continue
        else:
            flag = False
    print(">> High Priest: Bye my friend, and take care..")
