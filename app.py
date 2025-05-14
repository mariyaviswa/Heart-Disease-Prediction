import gradio as gr
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Mappings for readable inputs
sex_map = {"Male": 1, "Female": 0}
cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
fbs_map = {
    "Yes": 1,
    "No": 0
}
restecg_map = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
}
exang_map = {
    "Yes": 1,
    "No": 0
}
slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# Column order for prediction
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Prediction function
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):

    # Convert to numeric format for the model
    input_dict = {
        "age": age,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalach": thalach,
        "exang": exang_map[exang],
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal]
    }

    input_df = pd.DataFrame([input_dict], columns=feature_cols)

    # Prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction] * 100
    result_label = "🚨 Has Heart Disease" if prediction == 1 else "✅ No Heart Disease"
    confidence_str = f"{proba:.2f}%"

    # Create a display DataFrame
    display_df = pd.DataFrame({
        "Feature": [
            "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", "Fasting BS", "Resting ECG",
            "Max HR", "Exercise Angina", "Oldpeak", "Slope", "Major Vessels", "Thal"
        ],
        "Value": [
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]
    })

    # PDF generation
    tmp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(tmp_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("❤️ Heart Disease Prediction Report", styles['Title']))
    elements.append(Spacer(1, 12))

    table_data = [["Feature", "Value"]] + display_df.values.tolist()
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Prediction:</b> {result_label}", styles['Normal']))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence_str}", styles['Normal']))

    doc.build(elements)

    return display_df, result_label, confidence_str, tmp_pdf_path


# Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Radio(["Male", "Female"], label="Sex"),
    gr.Dropdown(["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], label="Chest Pain Type"),
    gr.Number(label="Resting Blood Pressure (trestbps)"),
    gr.Number(label="Cholesterol"),
    gr.Radio(["Yes", "No"], label="Fasting Blood Sugar > 120 mg/dl"),
    gr.Dropdown(["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], label="Resting ECG"),
    gr.Number(label="Maximum Heart Rate (thalach)"),
    gr.Radio(["Yes", "No"], label="Exercise Induced Angina"),
    gr.Number(label="Oldpeak (ST depression)"),
    gr.Dropdown(["Upsloping", "Flat", "Downsloping"], label="Slope"),
    gr.Slider(0, 3, step=1, label="Number of Major Vessels (ca)"),
    gr.Dropdown(["Normal", "Fixed Defect", "Reversible Defect"], label="Thal")
]

outputs = [
    gr.Dataframe(label="Input Summary", headers=["Feature", "Value"]),
    gr.Text(label="Prediction"),
    gr.Text(label="Confidence"),
    gr.File(label="Download PDF Report")
]

app = gr.Interface(fn=predict_heart_disease, inputs=inputs, outputs=outputs,
                   title="❤️ Heart Disease Prediction")

app.launch()
