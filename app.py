from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


model = joblib.load('LinearRegression1.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    data = {
        "name": request.form['name'],
        "company": request.form['company'],
        "year": int(request.form['year']),
        "kms_driven": int(request.form['kms_driven']),
        "fuel_type": request.form['fuel_type']
    }
    input_data = pd.DataFrame([data])
    
    prediction = model.predict(input_data)
    
    return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
