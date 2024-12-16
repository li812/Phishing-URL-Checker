import tensorflow as tf
import numpy as np  
import string
from keras.preprocessing import sequence
import urllib.parse
import re

# Load the model
loaded_model = tf.keras.models.load_model('models/model_40.keras')

def check_text_and_urls(text):
    # Define printable here
    printable = string.printable

    # Extract URLs
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    # Check each URL
    for url in urls:
        # Preprocess the URL
        url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
        X_input = sequence.pad_sequences(url_int_tokens, maxlen=75)
        
        # Get the probability of being malicious
        probability = loaded_model.predict(X_input)[0][0]
        
        # Display the result
        if probability > 0.5:
            result = "Malicious"
        else:
            result = "Safe"
        
        print("URL: ", url, "\nResult", f"The URL is predicted as: {result}\nProbability: {probability:.2f}")

# Test the function
check_text_and_urls("Here are kbhvhcdgfg   ggjhfghv fgjhfj   some URLs: https://www.google.com , kvjhvjvj,  http://xjbctcky.com/images/?app=com-d3&us.battle.net/login/en/?ref=http:  jhbjhgvghdfgdgf")