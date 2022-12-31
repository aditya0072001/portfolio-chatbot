import nltk
import pickle
import tensorflow
import numpy as np
import json
import random
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, Request
from pydantic import BaseModel

class Bot(BaseModel):
    input : str

bot = FastAPI()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
pad_sequences = pickle.load(open('pad_sequences.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
model = load_model('pchatbot')

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words =[lemmatizer.lemmatize(word) for word in sentence_words]
    return " ".join(sentence_words)

@bot.get("/")
async def root():
    return {"message": "Welcome to Portfolio Website Bot made for https://tripathiaditya.netlify.app/"}

@bot.post("/getResponse")
async def getRnformation(bot : Request):
    with open('intent.json') as content:
        data1 = json.load(content)

    tags = []
    inputs = []
    responses={}

    for intent in data1['intents']:
        responses[intent['tag']]=intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

    input_shape = 8
    texts_p = []
    req_info = await bot.json()
    prediction_input = clean_sentence(req_info["input"])
    

    #removing punctuation and converting to lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    #getting output from model
    output = model.predict(prediction_input)
    output = output.argmax()

    #finding the right tag and predicting
    response_tag = label_encoder.inverse_transform([output])[0]
    reponse_text = random.choice(responses[response_tag])

    return {
        "response_tag": response_tag,
        "response_text": reponse_text
    }







