from flask import  request, render_template, redirect, url_for, session, flash
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
import pickle
import numpy as np
import secrets

load_dotenv()
app = Flask(__name__)
app.secret_key = "your_secret_key"  # For session management

# MongoDB Configuration
app.config["MONGO_URI"] = os.getenv("MONGODB_URI")  # or hardcode your Mongo URI
mongo = PyMongo(app)

# Password hashing utility
bcrypt = Bcrypt(app)

# Load your models for prediction
scaler = pickle.load(open("model/standardScalar.pkl", "rb"))
model = pickle.load(open("model/modelForPrediction.pkl", "rb"))


# Route for dashboardpage
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/test_db')
def test_db():
    if mongo.db:  # Check if the database object is created
        return "MongoDB connected successfully!"
    else:
        return "Failed to connect to MongoDB."


@app.route('/result', methods=['GET'])
def result():
    # Get the prediction result from the session
    result = session.get('prediction_result', None)
    print(result)
    
    # If no result is found in the session, redirect to the dashboard
    if not result:
        flash('You need to make a prediction first!', 'error')
        return redirect(url_for('dashboard'))

    # Render the result page if the result exists
    rendered_result = render_template('result.html', result=result)

    # Clear the result from the session after rendering
    session.pop('prediction_result', None)

    return rendered_result


# Route for registering users
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if username exists
        existing_user = mongo.db.users.find_one({'username': username})

        if existing_user is None:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            mongo.db.users.insert_one({
                "username": username,
                "email": email,
                "password": hashed_password,
            })
            return redirect(url_for('dashboard'))
        else:
            flash('User already exists!', 'error')
    
    return render_template('register.html')

# Route for logging in
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user in the database
        user = mongo.db.users.find_one({'username': username})

        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid login credentials.', 'error')

    return render_template('login.html')

# Route for making a prediction
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        flash('Please log in to make predictions.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Extract input data
            Pregnancies = int(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            # Additional validation logic (e.g., check for negative values)
            if Glucose <= 0 or BloodPressure <= 0 or Age <= 0:
                flash('Values for Glucose, Blood Pressure, and Age must be greater than zero!', 'error')
                return redirect(url_for('dashboard'))

            # Make prediction
            data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)

            result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
            session['prediction_result'] = result


            # Save the prediction to MongoDB
            mongo.db.predictions.insert_one({
                "username": session['username'],
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
                "Age": Age,
                "Prediction": result,
            })

            flash(f'Prediction result: {result}', 'success')
            return redirect(url_for('result'))


        except ValueError as ve:
            flash(f'Value error: {str(ve)}', 'error')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('dashboard'))

    return render_template('dashboard.html')

#logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))  # Redirect to login page after logout

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)

