from flask import Flask, request, jsonify
import joblib
from main import preprocess_data
import pandas as pd


app = Flask(__name__)

model = joblib.load('./models/lgbm.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    preprocessed = preprocess_data(df)
    pre_columns = ['id', 'Fahrerlaubnis', 'Regional_Code', 'Vorversicherung', 'Vertriebskanal', 'Kundentreue', 'Female', 'Male', 'Age_20-29', 'Age_30-39', 'Age_40-49', 'Age_50-59', 'Age_60-69', 'Age_70-79', 'Age_80-89', '1-2_Year', 'lt_1_Year',
                   'gt_2_Years', 'Vorschaden_No', 'Vorschaden_Yes', 'Dominant_Premium', 'Jahresbeitrag_Log', 'Vertriebskanal_152.0', 'Vertriebskanal_26.0', 'Vertriebskanal_124.0', 'Vertriebskanal_160.0', 'Vertriebskanal_156.0', 'Kundentreue_scaled']
    for el in pre_columns:
        if el not in preprocessed.columns.tolist():
            preprocessed[el] = 0
    final_data = preprocessed.drop(columns=["Interesse", "Geschlecht", "Alter",
                                            "Alter_Fzg", "Vorschaden", "Jahresbeitrag", "Altersgruppen"])
    prediction = model.predict(final_data)

    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
