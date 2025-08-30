from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import pandas as pd

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

app = Flask(__name__)
app.secret_key = "supersecret"  # needed for session

@app.route('/')
def home():
    # Read prediction from session (if exists), then clear it
    prediction = session.pop("prediction_text", None)
    return render_template('index.html', prediction_text=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from form
        data = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
            "ocean_proximity": request.form["ocean_proximity"]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Transform & predict
        transformed = pipeline.transform(input_df)
        prediction = model.predict(transformed)

        # Save prediction in session
        session["prediction_text"] = f"Estimated Median House Value: ${prediction[0]:,.2f}"

    except Exception as e:
        session["prediction_text"] = f"Error: {str(e)}"

    # Redirect back to home (form will be empty, only prediction shown once)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True,port=5001)
