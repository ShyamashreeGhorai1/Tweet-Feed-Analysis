# importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import preprocess_kgptalkie as ps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



# importing dataset
df = pd.read_csv("twitter_training.csv")
df.rename(columns={'2401':'ids','Borderlands':'user','Positive':'target',
                     'im getting on borderlands and i will murder you all ,':'text'},inplace=True)
df = df[['target','text']]  



# Data Cleaning
def data_cleaning(df1):
  df1.dropna(inplace=True)
  df1.drop_duplicates(inplace=True)
  df1 = df1[df1['text'].apply(len)>1]
  return df1

data_cleaning(df)

df = df.drop(df[df['target'] == 'Irrelevant'].index) 
df['target'].value_counts().plot(kind='pie', autopct='%1.0f%%')
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)


# assigning value
df.loc[df['target'] == 'Positive','target'] = 1
df.loc[df['target'] == 'Negative','target'] = -1
df.loc[df['target'] == 'Neutral','target'] = 0



# Text Cleaning
# lowercase, remove url, html, punctuations, retweet
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: ps.remove_urls(x))
df['text'] = df['text'].apply(lambda x: ps.remove_html_tags(x))
df['text'] = df['text'].apply(lambda x: ps.remove_special_chars(x))
df['text'] = df['text'].apply(lambda x: ps.remove_rt(x))
df['target'] = df['target'].astype('int')



# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)



from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))



clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=list(stopwords))), ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
clf.fit(X_train, y_train)


# evaluation
from sklearn.metrics import accuracy_score
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
pd.crosstab(y_test, predictions)


# save model
import pickle

pickle.dump(clf, open('twitter_sentiment.pkl', 'wb'))