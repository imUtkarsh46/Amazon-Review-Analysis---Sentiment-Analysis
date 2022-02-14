
from PIL import Image
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import re
from afinn import Afinn
af = Afinn(language='en', emoticons=False, word_boundary=True)

# load save model
#cv = pickle.load(open("F:/ExcelR/Project 88/vectorizer.pkl", "rb"))
loaded_model = pickle.load(
    open('F:/ExcelR/Project 88/votingClassifier.pkl', 'rb'))
cv = pickle.load(open("F:/ExcelR/Project 88/vector.pkl", "rb"))

st.title("Welocme to Oximeter Review Analyzer App")

image = Image.open('F:/ExcelR/Project 88/banner01.jpg')
st.image(image)
st.text('       ')


input_review = st.text_area(
    "Hello!, User, Use of our app is very simple. You just to type your review or your can copy-paste any review from anywhere and put them into our box and click predict button. Our app will show you, your entered review is 'Negative' or 'Positive'")
st.text('       ')
st.text('       ')

# creating a function fot prediction


def text_cleaner(text):
    cleaned = re.sub('[^a-zA-Z]', " ", text)
    cleaned = cleaned.lower()
    cleaned = cleaned.split()
    cleaned = ' '.join(cleaned)
    return cleaned


if st.button("Predict"):
    cleaned_review = text_cleaner(input_review)
    cv_r = cv.transform([cleaned_review])
    result = loaded_model.predict(cv_r)
    if result == 1:
        #st.image("happy.png", width=70)
        st.header("Hey, its Positive Review !!!")
        st.header("Great Job!, Buddy")
    else:
        #st.image("sad.png", width=50)
        st.header("Sad, its Negative Review!")
        st.header("Your Tried Your Best!")
