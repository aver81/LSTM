# LSTM Next-Word Prediction App (Hamlet Dataset)

This is a Streamlit-based web application that uses LSTM models trained on Shakespeare's *Hamlet* to perform next-word prediction. Users can input a seed sentence and choose how many words they want the model to generate. The app also allows switching between 2 pre-trained RNN models.

---

## Features

- 📜 Predict the next N words from any given seed sentence
- 🧠 Choose between multiple LSTM models trained on the Hamlet dataset
- 🧰 Uses Keras + TensorFlow backend for prediction
- 🎨 Built with Streamlit for a clean web interface

## Project Structure
LSTM/

├── app.py # Main Streamlit app

├── model_v1.h5 # First trained LSTM model

├── tokenizer_v1.pkl # Corresponding tokenizer for LSTM base

├── model_v2.h5 # Second trained LSTM model

├── tokenizer_v2.pkl # Corresponding tokenizer for LSTM GRU

└── requirements.txt # Python dependencies
