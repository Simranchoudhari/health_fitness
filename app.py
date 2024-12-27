from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import joblib  # Import joblib to load the model

app = Flask(__name__)

# Load your data
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

# Prepare data (this part is based on your original code)
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Load the saved model
clf = joblib.load('model.pkl')  # Load your trained model

# Function to get bot response
def get_bot_response(user_input):
    # Example of processing the user input
    symptoms = user_input.split(',')  # Assuming symptoms are comma-separated
    input_vector = np.zeros(len(cols))

    # Convert user input symptoms to a vector
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in cols:
            index = np.where(cols == symptom)[0][0]
            input_vector[index] = 1

    # Make a prediction
    prediction = clf.predict([input_vector])
    predicted_disease = le.inverse_transform(prediction)[0]
    
    return f"You may have: {predicted_disease}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response = get_bot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
