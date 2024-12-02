
from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and column names
model = joblib.load('final_model.pkl')
col_names = joblib.load('column_names.pkl')  # Ensure this is a list of column names

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract form data
        feat_data = request.form.to_dict()
          
        # Optional: Convert data types as needed
        # For example, if your features are numeric:
        
        # Create DataFrame
        df = pd.DataFrame([feat_data])
        df = df.reindex(columns=col_names)  # Ensure correct column order
        
        # Make prediction
        prediction = model.predict(df)
        
        # Render the result template with prediction
        return render_template('after.html', data=prediction[0])  # Assuming single prediction
    except Exception as e:
        # Handle errors gracefully
        return render_template('after.html', data=f"Error: {str(e)}")

if __name__ == "__main__":

    
    app.run(debug=True)
