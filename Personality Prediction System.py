import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 2: Load dataset
df = pd.read_csv(r'C:\Users\Anirudhan\OneDrive\Desktop\mbti_1.csv.csv')

# Step 3: Basic Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)      # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_posts'] = df['posts'].apply(clean_text)

# Step 4: Simplify personality labels (e.g., Introvert vs Extrovert)
def get_label(personality):
    return 'Introvert' if personality[0] == 'I' else 'Extrovert'

df['label'] = df['type'].apply(get_label)

# Step 5: Vectorize the clean text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_posts'])
y = df['label']

# Step 6: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation Report:\n")
print(classification_report(y_test, y_pred))

# Step 9: User Input Prediction
print("\n🔮 Personality Prediction")
user_input = input("Enter a few sentences about yourself: ")
clean_input = clean_text(user_input)
input_vector = vectorizer.transform([clean_input])
prediction = model.predict(input_vector)

print(f"\n🧠 Predicted Personality: {prediction[0]}")
