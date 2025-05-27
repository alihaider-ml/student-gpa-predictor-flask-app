from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('gpa_predictor_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None,
                           study_hours='', sleep_hours='', social_hours='', physical_hours='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form (as integers)
        study_hours = int(request.form['study_hours'])
        sleep_hours = int(request.form['sleep_hours'])
        social_hours = int(request.form['social_hours'])
        physical_hours = int(request.form['physical_hours'])

        # Prepare input for prediction
        input_data = np.array([[study_hours, sleep_hours, social_hours, physical_hours]])
        prediction = model.predict(input_data)
        predicted_gpa = prediction[0]

        return render_template('index.html',
                               prediction_text=f"Predicted GPA: {predicted_gpa:.2f}",
                               study_hours=study_hours,
                               sleep_hours=sleep_hours,
                               social_hours=social_hours,
                               physical_hours=physical_hours)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}",
                               study_hours=request.form.get('study_hours', ''),
                               sleep_hours=request.form.get('sleep_hours', ''),
                               social_hours=request.form.get('social_hours', ''),
                               physical_hours=request.form.get('physical_hours', ''))

if __name__ == "__main__":
    app.run(debug=True)
