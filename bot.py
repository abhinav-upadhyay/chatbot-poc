#!/usr/bin/env python
import spacy
import random
import numpy as np
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

nlp = spacy.en.English()
gradboost = joblib.load('./gradboost.pkl')

greetings = ['Hello!', 'Hi!', 'Hey there!', 'How may I help you?']

def process_sentence(sentence):
    processed_words = []
    for word in sentence.split():
        if word.lower() == "i'm":
            processed_words.append("i")
            processed_words.append("am")
        elif word.lower() == "i'll":
            processed_words.append("i")
            processed_words.append("will")
        elif word[-1] == ',' or \
            word[-1] == '?' or \
            word[-1] == '.' or \
            word[-1] == '!': 
            word = word[:-1]
            processed_words.append(word)
        else:
            processed_words.append(word)
    return ' '.join(processed_words)

def extract_locations(sentence):
    locations = {'LOC':[], 'GPE':[]}
    doc = nlp((sentence))
    for ent in doc.ents:
        if ent.label_ == 'LOC':
            locations['LOC'].append(ent)
        elif ent.label_ == 'GPE':
            locations['GPE'].append(ent)
    return locations

def spacy_get_vec(sentence):
    vec = np.zeros(300)
    doc = nlp((sentence))
    for word in doc:
        vec += word.vector
    return vec

def get_location(sentence):
    locations = extract_locations(sentence)
    if len(locations['GPE']) == 0 and len(locations['LOC']) == 0:
        return ('where?', True)

    if len(locations['GPE']) == 0 and len(locations['LOC']) > 0:
        return ('sorry, %s is too big an area' % locations['LOC'][0], False)

    for location in locations['GPE']:
        return ('weather for %s is %d degrees' % (location, random.randint(0, 40)), False)

def greet(sentence):
    ind = random.randint(0, len(greetings) -1)
    return (greetings[ind], True)

def bye(sentence):
    return ('Have a nice day!', False)

def where(sentence):
    return ('have a nice trip!', False)


intent_actions = {'weather': get_location, 'greeting': greet, 'goodbye': bye, 'travel': where}

is_user_input_expected = True
original_intent = None
print(greetings[random.randint(0, len(greetings) - 1)]) 
while is_user_input_expected:
    s = input()
    if original_intent is None or original_intent == 'greeting':
        intent = gradboost.predict(spacy_get_vec(s))[0]
        original_intent = intent
    action = intent_actions[intent]
    bot_response, is_user_input_expected = action(s)
    print(bot_response)
