import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
random_forest_model = joblib.load(r'random_forest_model.joblib')
linear_regression_model = joblib.load(r'linear_regression_model.joblib')

@app.route("/recommend", methods=["POST"])
def predict():
    data = request.json

    # Extract input values from the JSON payload
    accelerometer = data.get("accelerometer")
    microphone = data.get("microphone")
    pressure = data.get("pressure")
    wifi = data.get("wifi")
    vibrator = data.get("vibrator")
    touch = data.get("touch")
    proximity = data.get("proximity")
    magnetic = data.get("magnetic")
    light = data.get("light")
    gyroscope = data.get("gyroscope")
    gravity = data.get("gravity")
    gps = data.get("gps")
    flashlight = data.get("flashlight")
    battery = data.get("battery")
    buttons = data.get("buttons")
    display = data.get("display")
    bluetooth = data.get("bluetooth")
    frequency = data.get("frequency")

    random_forest_input_data = pd.DataFrame([[
        accelerometer, microphone, pressure, wifi, vibrator, touch,
        proximity, magnetic, light, gyroscope, gravity, gps,
        flashlight, battery, buttons, display, bluetooth
    ]], columns=[
        'accelerometer', 'microphone', 'pressure', 'wifi', 'vibrator',
        'touch', 'proximity', 'magnetic', 'light', 'gyroscope',
        'gravity', 'gps', 'flashlight', 'battery', 'buttons',
        'display', 'bluetooth'
    ])

    random_forest_predicted_error = random_forest_model.predict(random_forest_input_data)
    random_forest_result = int(random_forest_predicted_error[0])

    linear_regression_input_data = pd.DataFrame({
        'error': [random_forest_result],
        'frequency': [frequency]
    })

    linear_regression_recommendation = linear_regression_model.predict(linear_regression_input_data)
    linear_regression_result = round(linear_regression_recommendation[0])
    return jsonify({"recommendation": linear_regression_result})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)  # Use '0.0.0.0' to allow external connections
