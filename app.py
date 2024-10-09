import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, accuracy_score, roc_auc_score
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('spam.csv', encoding='Latin-1')
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25)

# Create Naive Bayes Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('nb', MultinomialNB())  
])

# Train the model
clf.fit(X_train, y_train)

# Streamlit Application
st.title('Spam Detection App')

st.write("""
### Enter a message to check if it's spam or ham:
""")

# User input for email text
email_text = st.text_input("Enter your message here", "")

# Prediction button
if st.button('Predict'):
    if email_text != "":
        prediction = clf.predict([email_text])[0]
        if prediction == 0:
            st.success("This is a Ham Email!")
        else:
            st.warning("This is a Spam Email!")

# Show model evaluation metrics
st.write("""
### Model Evaluation
""")
if st.checkbox('Show Confusion Matrix'):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

if st.checkbox('Show ROC Curve'):
    pred_prob_test = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pred_prob_test)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, pred_prob_test))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Show Classification Report
if st.checkbox('Show Classification Report'):
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
