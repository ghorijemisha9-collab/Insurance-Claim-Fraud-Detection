from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================
# Load Model & Preprocessor
# ==========================
model = joblib.load("../models/fraud_model.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        # ==========================
        # Read Form Data
        # ==========================

        input_data = {
            "Month": request.form["Month"],
            "WeekOfMonth": int(request.form["WeekOfMonth"]),
            "DayOfWeek": request.form["DayOfWeek"],
            "Make": request.form["Make"],
            "AccidentArea": request.form["AccidentArea"],
            "DayOfWeekClaimed": request.form["DayOfWeekClaimed"],
            "MonthClaimed": request.form["MonthClaimed"],
            "WeekOfMonthClaimed": int(request.form["WeekOfMonthClaimed"]),
            "Sex": request.form["Sex"],
            "MaritalStatus": request.form["MaritalStatus"],
            "Age": int(request.form["Age"]),
            "Fault": request.form["Fault"],
            "PolicyType": request.form["PolicyType"],
            "VehicleCategory": request.form["VehicleCategory"],
            "VehiclePrice": request.form["VehiclePrice"],
            "RepNumber": int(request.form["RepNumber"]),
            "Deductible": int(request.form["Deductible"]),
            "DriverRating": int(request.form["DriverRating"]),
            "Days_Policy_Accident": request.form["Days_Policy_Accident"],
            "Days_Policy_Claim": request.form["Days_Policy_Claim"],
            "PastNumberOfClaims": request.form["PastNumberOfClaims"],
            "AgeOfVehicle": request.form["AgeOfVehicle"],
            "AgeOfPolicyHolder": request.form["AgeOfPolicyHolder"],
            "PoliceReportFiled": request.form["PoliceReportFiled"],
            "WitnessPresent": request.form["WitnessPresent"],
            "AgentType": request.form["AgentType"],
            "NumberOfSuppliments": request.form["NumberOfSuppliments"],
            "AddressChange_Claim": request.form["AddressChange_Claim"],
            "NumberOfCars": request.form["NumberOfCars"],
            "Year": int(request.form["Year"]),
            "BasePolicy": request.form["BasePolicy"]
        }

        # ==========================
        # Convert to DataFrame
        # ==========================

        df = pd.DataFrame([input_data])

        # ==========================
        # Preprocess
        # ==========================

        x = preprocessor.transform(df)

        # ==========================
        # Prediction
        # ==========================

        prediction = model.predict(x)[0]

        probability = model.predict_proba(x)[0][1]

        if prediction == 1:
            result = "⚠ Fraud Claim"
        else:
            result = "✅ Genuine Claim"

        return render_template(
            "index.html",
            prediction_text=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return f"Error : {e}"


if __name__ == "__main__":
    app.run(debug=True)