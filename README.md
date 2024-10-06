EMAIL SPAM DETECTION

![20945480](https://user-images.githubusercontent.com/118047264/226919729-d56fbf6c-ce33-41cb-b2ae-dad1141238e7.jpg)

Spam Mail Detection with Machine Learning

Spam emails can be a major nuisance, but machine learning offers a powerful way to filter them out automatically. This project demonstrates how to build a spam detection model using Python and deploy it as a web application with Streamlit.

Project Overview

Dataset: SMS Spam Collection Dataset from Kaggle (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code)
Packages: pandas for data manipulation, scikit-learn for machine learning, matplotlib/seaborn for data visualization, and Streamlit for web app deployment.
Process: Data preprocessing, NLP techniques, text classification using Multinomial Naive Bayes (MNB), model evaluation, and Streamlit deployment.
Model Description

The core of this project is a Multinomial Naive Bayes (MNB) classifier, well-suited for text classification based on word frequency. It analyzes the frequency of words in emails and their association with spam/ham labels to predict whether a new email is likely spam or not.

Code Breakdown:

1. Data Loading and Exploration

Import libraries (pandas, scikit-learn, matplotlib)
Load the spam dataset using pandas
Explore data shape, information, basic statistics
Perform data cleaning (missing values, duplicates)
Visualize spam distribution

2. Data Preprocessing

Create a new target variable ("Spam") based on the "Category" column
Generate a WordCloud visualization to explore frequently used words in spam emails

3. Feature Engineering

Apply CountVectorizer to convert text messages into numerical features based on word frequency

4. Model Training and Evaluation

Split the data into training and testing sets
Implement a evaluate_model function that:
Fits the MNB model to the training data
Predicts labels for training and testing sets
Computes and displays evaluation metrics (confusion matrix, ROC curve, classification report)

5. Spam Detection Function:

Create a detect_spam function that takes an email message as input
Uses the trained MNB model to predict whether the email is spam or not

6. Streamlit Deployment

Create a Streamlit app using streamlit.io
Display a user-friendly interface with a text box for entering email messages
When a user submits an email, call the detect_spam function and display the prediction

Example Usage:

Python
sample_email = 'Free Tickets for IPL'
result = detect_spam(sample_email)
print(result)  # Output: "This is a Spam Email!"

Use code with caution.

Streamlit App Example:

Python
import streamlit as st

# Load the trained model (replace with your model loading logic)
model = ...  # Load your trained MNB model from disk

def detect_spam(email_text):
    prediction = model.predict([email_text])
    if prediction == 0:
        return "This is a Ham Email!"
    else:
        return "This is a Spam Email!"

st.title("Spam Email Detection")
email_text = st.text_input("Enter an email message:")

if email_text:
    result = detect_spam(email_text)
    st.write(result)
Use code with caution.

Deployment Instructions:

Install Streamlit: pip install streamlit
Create a Python file (app.py) with the Streamlit app code.
In the terminal, navigate to the directory containing app.py and run: streamlit run app.py
Open http://localhost:8501 in your web browser to access the app.
Key Takeaways:

This project demonstrates a practical approach to spam detection using machine learning and its deployment as a web application. Explore the code for a deeper understanding and make adjustments to improve the model's performance. By understanding the MNB model and its application, you can delve into further advancements in text classification and spam filtering.
