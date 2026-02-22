from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features = [np.array(input_features)]
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Person is likely to have Diabetes"
    else:
        result = "Person is unlikely to have Diabetes"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    #Because when you deploy on AWS, it must accept external requests. This prevents future confusion.