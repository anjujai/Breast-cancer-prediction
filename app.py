from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('model.pkl', 'rb'))
# flask application
app = Flask(__name__)
# this line re routes and load the index.html template on the web
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict_new', methods=['POST'])
def predict():
    
    # Getting all the features given by the user on frontend form
    features = request.form['feature']
    
    # Splitting the features
    features = features.split(',')

    # Converting the features to a numpy array
    np_features = np.asarray(features, dtype=np.float32)

    # Doing the prediction of the model
    pred = model.predict(np_features.reshape(1, -1))
    
    # Display the message using list comprehension
    message = ['Cancerous' if pred[0] == 1 else 'Not Cancerous']
    
    # Render the template with the prediction message
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)