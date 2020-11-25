from flask import Flask, request, make_response, jsonify
from flask_restplus import Api, Resource, fields
import numpy as np
import pandas as pd
import pickle

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from text_summarization import generate_summary

flask_app = Flask(__name__)

@flask_app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    flask_app.run()

app = Api(app = flask_app,  
		  version = "1.0", 
		  title = "Trial", 
		  description = "Predict") 

classifier_sentiment = pickle.load(open('model_nlp.pkl', 'rb'))        

cv_sentiment = pickle.load(open('cv_sentiment.pkl', 'rb'))        

name_space_sentiment = app.namespace('sentiment', description='Prediction APIs')

model_sentiment = app.model('Prediction params', 
				  {'review': fields.String(required = True, 
				  							   description="Reviews", 
    					  				 	   help="Reviews cannot be blank")})

name_space_summary = app.namespace('summary', description='Prediction APIs')

model_summary = app.model('Prediction params', 
				  {'content': fields.String(required = True, 
				  							   description="Content", 
    					  				 	   help="Content cannot be blank")},
                {'sentences': fields.String(required = True, 
				  							   description="Number of sentences", 
    					  				 	   help="Number of sentences cannot be blank")})

classifier_mood = pickle.load(open('model_mood.pkl', 'rb'))        

cv_mood = pickle.load(open('cv_mood.pkl', 'rb'))      

name_space_mood = app.namespace('mood', description='Prediction APIs')

model_mood = app.model('Prediction params', 
				  {'text': fields.String(required = True, 
				  							   description="Text", 
    					  				 	   help="Text cannot be blank")})                                               

@name_space_sentiment.route("/")
class SentimentClass(Resource):

    def options(self):
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response

    @app.expect(model_sentiment)		
    def post(self):
            try:
                    data = request.json
                    review = re.sub('[^a-zA-Z]', ' ', data['review'])
                    review = review.lower()
                    review = review.split()
                    ps = PorterStemmer()
                    all_stopwords = stopwords.words('english')
                    all_stopwords.remove('not')
                    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
                    review = ' '.join(review)
                    X_test = cv_sentiment.transform([review]).toarray()
                    classification = classifier_sentiment.predict(X_test)
                    response = jsonify({
                        "result": str(classification[0])
                    })
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response
            except Exception as error:
                    return jsonify({
                        "result": "Could not make prediction",
                        "error": str(error)
                    })


@name_space_summary.route("/")
class SummaryClass(Resource):

    def options(self):
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response

    @app.expect(model_summary)		
    def post(self):
            try:
                    data = request.json
                    classification = generate_summary(data['content'], int(data['sentences']))
                    classification = classification + "."
                    response = jsonify({
                        "result": classification
                    })
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response
            except Exception as error:
                    return jsonify({
                        "result": "The number of sentences in the Summary cannot be greater than the content. Try again with lesser number of sentences.",
                        "error": str(error)
                    })

@name_space_mood.route("/")
class MoodClass(Resource):

    def options(self):
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response

    @app.expect(model_mood)		
    def post(self):
            try:
                    data = request.json
                    text = re.sub('[^a-zA-Z]', ' ', data['text'])
                    text = text.lower()
                    text = text.split()
                    ps = PorterStemmer()
                    all_stopwords = stopwords.words('english')
                    all_stopwords.remove('not')
                    text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
                    text = ' '.join(text)
                    X_test = cv_mood.transform([text]).toarray()
                    classification = classifier_mood.predict(X_test)
                    response = jsonify({
                        "result": str(classification[0])
                    })
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response  
            except Exception as error:
                    return jsonify({
                        "result": "Could not make prediction",
                        "error": str(error)
                    })