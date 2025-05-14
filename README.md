# Heart Disease Prediction

This project focuses on developing a machine learning-based web application to predict the presence of heart disease in patients based on various medical attributes. Using the Random Forest Classifier, the model is trained on the well-known UCI Heart Disease dataset. The trained model is then integrated with an interactive Gradio UI, allowing users to input patient details such as gender, chest pain type, fasting blood sugar level, and more in a user-friendly format (e.g., using dropdowns and yes/no switches).

The application processes the inputs, converts them into the required numeric format, and predicts whether the patient is likely to have heart disease. It also shows the model's confidence (accuracy score) and provides an option to download a PDF report summarizing the input values and prediction result. The goal is to create an accessible tool for preliminary risk assessment in a healthcare setting, demonstrating how machine learning can support medical decision-making.

This project not only highlights core machine learning techniques like data preprocessing, model training, and evaluation, but also demonstrates how to deploy ML models in real-world applications using Gradio, with a strong emphasis on usability, clarity, and result interpretability.

## About the dataset

Dataset taken from Kaggle. Here also I provided the dataset named `heart.csv`.<br>
Also you can download from <a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Here.</a>

## Setup Process

### step1:
Install all the dependencies from `requirements.txt`.<br />
<pre>
    pip install -r requirements.txt
</pre>

### step2:

#### Method1:
If you want to build the model from scratch or undestand the dataset with visualization then run the 
`notebook.ipynb` directly, It will save the model in your local system. Then using the model run the `app.py` file

#### Method2:
You can run the `app.py` file directly with the help of pretrained model named `heart_disease_model.pkl`


## Additional Information 

To understand the dataset attributes, One text file provided with named ->  `keynotes.txt`

## User Interface 

