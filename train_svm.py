from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle

# Sample training data (can be expanded with real labels)
X_train = [
    "Hello, how are you?",
    "What are your products?",
    "I want to make a complaint.",
    "Thank you for your help.",
    "Can you recommend something?",
    "I have a problem with my order."
]
y_train = ["greet", "product_query", "complaint", "thanks", "recommend", "problem"]

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectors, y_train)

# Save the model and vectorizer
with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(svm_model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("SVM model and vectorizer saved!")