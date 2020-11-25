import numpy as np
import pandas as pd
import pickle

df_train = pd.read_csv("train.txt",sep=';',names=['text','emotion'])

y_train = df_train.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 16000):
    text = re.sub('[^a-zA-Z]', ' ', df_train['text'][i])
    text = text.lower() 
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
    text = ' '.join(text)
    corpus.append(text)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X_train = cv.fit_transform(corpus).toarray()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=16947, random_state=0) #16947 is the number of features
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('model_mood.pkl','wb'))
pickle.dump(cv, open('cv_mood.pkl','wb'))
