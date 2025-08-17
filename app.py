import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_option = st.selectbox("Choose a model:", ["LSTM", "GRU"])

if model_option == "LSTM":
    model = load_model("model.h5")
elif model_option == "GRU":
    model = load_model("GRU model.h5")

with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

max_sequence_len = model.input_shape[1] + 1

def next_n_word_predictions(text,n_words):
    for _ in range(n_words):
        tokenized_sentence = tokenizer.texts_to_sequences([text])[0]
        padded_tokenized_sentence = pad_sequences([tokenized_sentence],maxlen=max_sequence_len-1,padding='pre')
        predicted_probs = model.predict(padded_tokenized_sentence,verbose=0)
        predicted_index = np.argmax(predicted_probs)

        output = ""
        for word,index in tokenizer.word_index.items():
            if index==predicted_index:
                output = word
                break
        text = text + " " + output
    return text



st.title('LSTM N-Word Prediction with Dropout')

text = st.text_input('Enter the sentence:',"to be or not to be")
n_words = int(st.number_input('Enter how many words you want to predict: ',3))

if st.button('Predict'):
    predicted_words = next_n_word_predictions(text,n_words)
    st.text_area(f'📘 Predicted text (next {n_words} words):', predicted_words)


