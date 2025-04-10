import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
questions = [
    "Hello",
    "What are your working hours?",
    "How can I reset my password?",
    "Where is my order?",
    "How do I return an item?",
    "Can I change my shipping address?",
    "Thank you"
]

answers = [
    "Greetings of the day!! How can I help you?",
    "Our working hours are 9 AM to 5 PM, Monday to Friday.",
    "To reset your password, click on 'Forgot Password' at login.",
    "You can track your order in the 'My Orders' section.",
    "To return an item, visit your order and click 'Return'.",
    "Yes, you can change your shipping address before the order is shipped.",
    "Have a Nice day!!"
]

# Preprocess and vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train model
model = MultinomialNB()
model.fit(X, range(len(answers)))

# Chatbot function
def chatbot_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    pred = model.predict(user_input_vector)[0]
    return answers[pred]

# Test the chatbot
while True:
    user_query = input("You: ")
    if user_query.lower() in ['exit', 'quit']:
        break
    response = chatbot_response(user_query)
    print("Bot:",response)