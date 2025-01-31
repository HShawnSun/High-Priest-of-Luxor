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
from sklearn.linear_model import LogisticRegression
from util import text_preprocessing, correct

# Use absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'dataset')

X_train_deity = [
# Amun-Ra
"The god I worship is Amun-Ra, the giver of strength and wisdom.",
"Amun-Ra, guide my leadership with your divine authority.",
"I seek Amun-Ra's blessing to rule justly and with power.",
"Amun-Ra, grant me the strength to overcome all challenges.",
"I place my trust in Amun-Ra to bring prosperity to my kingdom.",
# Amun-Ra without the word "Amun-Ra"
"I seek the strength to lead my people with wisdom and fairness.",
"Bless me with the authority to govern and protect my kingdom.",
"Guide my actions, so I may rule justly and with honor.",
"Grant me the ability to unite those under my rule and build a strong nation.",
"I desire the power to overcome any challenges that threaten my reign.",
"Provide me with the courage to make difficult decisions for the good of my people.",
"Bless my leadership with prosperity and peace for my kingdom.",
"Guide me in maintaining order and stability in my domain.",
"May my rule be marked by progress, wisdom, and respect.",
"I pray for your favor as I navigate the responsibilities of leadership.",

# Osiris
"My god is Osiris.",
"I worship Osiris.",
"Osiris is the deity I follow.",
"I seek Osiris' blessing for my harvest.",
"Osiris guides the cycles of life and death.",
# Osiris without the word "Osiris"
"Please bless my crops to grow abundant and strong this season.",
"Grant my farm the nourishment it needs to prosper despite difficult weather.",
"There is famine in my village, I want to pray for blessings of good harvests.",
"May the earth yield its fruits, and may my labor be rewarded with plenty.",
"Bless the soil that it may be fertile and yield abundant crops for the coming year.",
"I pray for the renewal of the land, that it may restore life to our people.",
"Help us to cultivate the land and honor the cycles of nature.",
"May our crops grow strong and our animals healthy to sustain us.",
"I offer thanks for the food that sustains us, and I pray for more to feed the hungry.",
"Guide the growth of our fields and bless us with an abundant harvest.",

# Anubis
"My god is Anubis.",
"I worship Anubis.",
"Anubis is the deity I follow.",
"I seek Anubis' guidance.",
"Anubis is my protector.",
# Anubis without the word "Anubis"
"I ask for your protection in the afterlife and safe passage to the realm of the dead.",
"Guide my ancestors' souls to their eternal rest and protect them on their journey.",
"Bless the preparations for my loved one's burial, that they may be honored properly.",
"May the mummification process be done with care, so the soul may be preserved forever.",
"I seek your blessing for the peaceful transition of the departed into the afterlife.",
"Protect the remains of the deceased, so they may be free from desecration.",
"Grant us comfort as we lay our loved ones to rest, knowing they are under your care.",
"May the journey of the soul be smooth, and may they find peace in the afterlife.",
"Bless the tomb and the rites performed for the departed, that they may be properly honored.",
"Guide me as I prepare for my own transition, that I may walk with you into eternity.",

# Thoth
"My god is Thoth, the god of wisdom and knowledge.",
"I worship Thoth, the divine scribe and teacher.",
"Thoth guides me in my pursuit of knowledge and understanding.",
"I seek Thoth's blessing in my studies and intellectual pursuits.",
"Thoth is the deity I follow.",
# Thoth without the word "Thoth"
"Grant me wisdom and understanding to succeed in my studies.",
"Help me to remember and apply what I have learned in my pursuit of knowledge.",
"Bless me with the clarity to solve problems and gain insight into the mysteries of life.",
"Guide me in my search for knowledge, so I may better myself and help others.",
"May I have the patience to study diligently and the intelligence to comprehend complex ideas.",
"I pray for guidance in my education, that I may achieve success in my endeavors.",
"Help me to understand the truth and gain the knowledge needed to navigate the world.",
"Grant me the ability to pass my exams and move forward in my academic journey.",
"I seek your assistance in learning the arts and sciences, so I may make meaningful contributions to society.",
"May my studies lead me to greater understanding and the ability to serve others.",

# Sekhmet
"My god is Sekhmet.",
"I worship Sekhmet.",
"Sekhmet is the deity I follow.",
"I seek Sekhmet's strength in battle.",
"Sekhmet guides me with courage and power.",
# Sekhmet without the word "Sekhmet"
"Grant me the strength to defend myself and my loved ones when the time comes.",
"Bless me with the courage to face challenges head-on, no matter how fierce.",
"Guide me in my battles, that I may win with honor and strength.",
"Grant me the power to overcome my enemies and protect my people.",
"May my warriors be strong and fearless as they fight for our cause.",
"Bless me with the ability to fight both for survival and for justice.",
"I seek your strength in times of conflict, to remain resolute and unwavering.",
"Help me balance aggression and compassion, finding wisdom in the heat of battle.",
"May I always fight for what is right, with your power guiding me.",
"Guide me in understanding the duality of war and peace, and when each must be embraced.",

# Hathor
"My god is Hathor.",
"I worship Hathor.",
"Hathor is the deity I follow.",
"I seek Hathor's blessing for love and harmony.",
"Hathor guides my relationships and family.",
# Hathor without the word "Hathor"
"Bless my relationship with love and harmony, so we may grow together.",
"Grant me the joy of a loving partnership, free from strife or sorrow.",
"I pray for the health and happiness of my children, that they may grow strong and wise.",
"Please bless me with the ability to nurture love in my home and relationships.",
"Grant me the blessings of fertility, that I may be a parent and guide my children well.",
"Bless my relationship with passion and understanding, helping us to grow closer.",
"Guide my family with compassion and understanding, especially in times of difficulty.",
"I pray for peace and love in my household, that our bonds may always remain strong.",
"Grant me the grace to be a loving and supportive partner, and may our love flourish.",
"May my children grow in health, love, and wisdom, and may I be a wise guide to them.",
]

y_train_deity = ['Amun-Ra']*15 + ['Osiris']*15 + ['Anubis']*15 + ['Thoth']*15 + ['Sekhmet']*15 + ['Hathor']*15

# Initialize vectorizer and classifier
vectorizer = CountVectorizer()
X_train_deity_vectors = vectorizer.fit_transform(X_train_deity)
classifier_deity = LogisticRegression()
classifier_deity.fit(X_train_deity_vectors, y_train_deity)

def get_deity(user_input):
    deities = ['Amun-Ra', 'Osiris', 'Anubis', 'Thoth', 'Sekhmet', 'Hathor']
    for deity in deities:
        if deity in user_input:
            return deity
    user_input = [correct(i) for i in user_input.split(' ')]
    user_input = (' ').join(user_input)
    user_input_vector = vectorizer.transform([user_input])
    deity = classifier_deity.predict(user_input_vector)[0]
    return deity