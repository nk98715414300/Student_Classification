from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

message = [
    "win a free prize now",
    "Congratulations you won money",
    "Claim your free gift",
    "Get free cash today",
    "You won a lottery",
    "Free offer click now",
    "Complete your college assignment",
    "Tomorrow is your exam",
    "Please send me your notes",
    "Lets study machine learning",
    "Your class starts at 9 AM",
    "Please submit your project"
]

labels = [
    "Spam", "Spam", "Spam",
    "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam",
    "Not Spam", "Not Spam", "Not Spam"
]

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(message, labels)

print("Spam Classifier is Ready!")

new_message = input("Enter your message: ")

prediction = model.predict([new_message])

print("Prediction:", prediction[0])

probabilities = model.predict_proba([new_message])[0]

for label, probability in zip(model.classes_, probabilities):
    print(label, ":", round(probability * 100, 2), "%")