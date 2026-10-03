
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("pcb_defect_model.pkl")

print("PCB Defect Detection Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    new_pcb = pd.DataFrame([{
        "Voltage_V": float(data["Voltage_V"]),
        "Current_A": float(data["Current_A"]),
        "Resistance_Ohm": float(data["Resistance_Ohm"]),
        "Temperature_C": float(data["Temperature_C"]),
        "Solder_Quality": float(data["Solder_Quality"]),
        "Component_Alignment": float(data["Component_Alignment"]),
        "Track_Quality": float(data["Track_Quality"]),
        "Solder_Type": data["Solder_Type"],
        "Inspection_Status": data["Inspection_Status"]
    }])

    # Make prediction
    prediction = model.predict(new_pcb)

    # Get prediction probabilities
    probabilities = model.predict_proba(new_pcb)

    defective_probability = probabilities[0][1] * 100
    non_defective_probability = probabilities[0][0] * 100

    if prediction[0] == 1:
        result = "Defective"
    else:
        result = "Non-Defective"

    return jsonify({
        "prediction": result,
        "defective_probability": round(defective_probability, 2),
        "non_defective_probability": round(non_defective_probability, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)