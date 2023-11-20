import streamlit as st
from textblob import TextBlob
import re
import nltk
import preprocess_kgptalkie as ps
import streamlit as st

# loading the saved model
import pickle


def model_prediction(value):
  loaded_model = pickle.load(open('twitter_sentiment.pkl','rb'))
  result = ''
  prediction = loaded_model.predict(value)
  if prediction == 1:
     result = 'Positive'
  elif prediction == -1:
     result = 'Negative'
  elif prediction == 0:
     result = 'Neutral'
  return result



# Create a function to clean the tweets
def cleanTxt(text):
 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
 text = re.sub('#', '', text) # Removing '#' hash tag
 text = re.sub('RT[\s]+', '', text) # Removing RT
 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink

 return text

# Create a function to get the subjectivity
def getSubjectivity(text):
  text = cleanTxt(text)
  subjectivity = TextBlob(text).sentiment.subjectivity
  return subjectivity

# Create a function to get the polarity
def getPolarity(text):
  text = cleanTxt(text)
  polarity = TextBlob(text).sentiment.polarity
  return  polarity




def main():
  sentiment = ''
  polarity = ''
  subjectivity = ''
  # giving a title
  st.title("Tweet Feed analyzer")
  # getting input from the user
  inputText = [st.text_input("Enter the text")]
  print(inputText)

  # creating a button for showing the result
  if st.button('Result'):
    print(inputText, type(inputText))
    sentiment = model_prediction(inputText)
    polarity = round(getPolarity(inputText[0]),2)
    subjectivity = round(getSubjectivity(inputText[0]),2)
    

    st.text(f"Sentiment is: {sentiment}")
    st.text(f"Polarity is:  {polarity}")
    st.text(f"Subjectivity is: {subjectivity}")
  
    



if __name__ == '__main__':
  main()

  






