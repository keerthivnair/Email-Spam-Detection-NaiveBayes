import joblib
import streamlit as st

model = joblib.load('spam_classifier.pkl')
cv = joblib.load('vectorizer.pkl')

st.title("📧 Email Spam Classifier")
st.markdown("Enter an email or message below to check if it's **Spam** or **Not Spam**.")

user_input = st.text_area("Your email or message here:", height=200)

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter some text to classify')
    else:
        user_input_count = cv.transform([user_input])
        pred = model.predict(user_input_count)

        if pred[0] == 1:
            st.error('This message is **SPAM**')
        else:
            st.success("This is **NOT SPAM**.")