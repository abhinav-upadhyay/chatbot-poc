#!/usr/bin/env python
import spacy
import random
import numpy as np
from sklearn.externals import joblib
import warnings
from stopwords import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.en.English()
intent_analyzer = joblib.load('./intent.pkl')
question_analyzer = joblib.load('./question.pkl')
sentiment_analyzer = joblib.load('./sentiment.pkl')
class_analyzer = joblib.load('./class.pkl')
is_question_analyzer = joblib.load('./is_question.pkl')
user_name = None
greeting_responses = ['How may I help you?', 'What can I do for you?', 'How can I be of service to you?', 'My name is R2D2, How can I help you?']
greetings = ['Hi!', 'Hey there!', 'Hello', 'Aloha!', 'Hey!']
user_name = None

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

def extract_name(sentence):
    doc = nlp((sentence))
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent
    return None

def spacy_get_vec(sentence, filter_stopwords=False):
    vec = np.zeros(300)
    doc = nlp((sentence))
    for word in doc:
        if filter_stopwords:
            if word.lower_ in ENGLISH_STOP_WORDS:
                continue
        vec += word.vector
    return vec

def get_sentiment_vec(sentence):
    vec = np.zeros(600)
    doc = nlp((sentence))
    for i,word in enumerate(doc):
        if i < len(doc) -1:
            tempvec = np.append(word.vector, doc[i + 1].vector)
            vec += tempvec
        else:
            tempvec = np.append(word.vector, np.zeros(300))
            vec += tempvec
    return vec

def weather(sentence):
    locations = extract_locations(sentence)
    if len(locations['GPE']) == 0 and len(locations['LOC']) == 0:
        return ('Where?', True, weather)

    if len(locations['GPE']) == 0 and len(locations['LOC']) > 0:
        return ('sorry, %s is too big an area' % locations['LOC'][0], True, weather)

    for location in locations['GPE']:
        return ('weather for %s is %d degrees' % (location, random.randint(0, 40)), True, None)


def greet(sentence):
    ind = random.randint(0, len(greeting_responses) -1)
    return (greeting_responses[ind], True, None)

def bye(sentence):
    return ('Have a nice day!', False, None)

def where(sentence):
    return ('have a nice trip!', False, None)

def users_name(sentence):
    global user_name
    name = extract_name(sentence)
    if user_name is None and name is None:
        return ('You didn\'t tell me your name', True, users_name)
    if name is None and user_name is not None:
        return ('Hi %s :-)' % user_name, True, None)
    user_name = name
    return ('Hello %s :-D' % name, True, None)

def sorry(sentence):
    responses = ['Too bad :(', 'I am very sorry to hear that.', 'I feel sorry to know that :-(', 'I am sorry']
    return (responses[random.randint(0, len(responses) -1)], True, None)

def name(sentence):
    return ('My name is R2D2', True, None)

def marriage(sentence):
    responses = ['Romance is not for me', 'I don\'t have any girlfriend', 'I am too young for that']
    return (responses[random.randint(0, len(responses) -1)], True, None)

def congrats(sentence):
    responses = ['Congrats!', 'Happy to hear that :-)', 'Wow! :-D']
    return (responses[random.randint(0, len(responses) -1)], True, None)

def bot(sentence):
    return 'I am a bot', True, None

def neutral(sentence):
    responses = ['Oh, ok!', 'Hmm', 'Ok', 'Alright', 'Ah', 'Oh']
    return (responses[random.randint(0, len(responses) -1)], True, None)

def bot_location(sentence):
    responses = ['I live in the cloud', 'I live on the server', 'As we speak I am replicating myself over the internet']
    return (responses[random.randint(0, len(responses) -1)], True, None)

def greet_response(sentence):
    responses = ['I am good, thanks!', 'Pretty good, thanks for asking %s' %
            (user_name if user_name is not None else '') , 'I am Ok', 'Everything is fine']
    return (responses[random.randint(0, len(responses) -1)], True, None)


intent_actions = {'weather': weather, 'greeting': greet, 'goodbye': bye, 'travel': where}
sentiment_actions = {'sorry': sorry, 'congrats': congrats, 'neutral': neutral}
question_actions = {'users_name': users_name, 'name': name, 'marriage': marriage,
        'bot': bot, 'location': bot_location, 'greet_response': greet_response }

is_user_input_expected = True
original_intent = None
print(greetings[random.randint(0, len(greetings) - 1)]) 
action = None
while is_user_input_expected:
    s = input()
    s = s.strip()
    if s == '':
        continue
    vec = spacy_get_vec(s)
    if action is not None:
        bot_response, is_user_input_expected, action = action(s)
        print(bot_response)
        continue

    is_intent = class_analyzer.predict(spacy_get_vec(s, True))[0] == 'intent'
    if not is_intent:
        is_question = is_question_analyzer.predict(vec)[0] == 'question'
        if is_question:
            question = question_analyzer.predict(vec)[0]
            action = question_actions[question]
        else:
            sentiment = sentiment_analyzer.predict(get_sentiment_vec(s))[0]
            action = sentiment_actions[sentiment]
        bot_response,is_user_input_expected,action = action(s)
        print(bot_response)
        continue
    intent = intent_analyzer.predict(vec)[0]
    action = intent_actions[intent]
    bot_response, is_user_input_expected,action = action(s)
    print(bot_response)
