!pip install imbalanced-learn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


nltk.download('stopwords')


class LdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.lda = LatentDirichletAllocation(n_components=self.n_components, random_state=self.random_state)
        self.vectorizer = CountVectorizer(max_features=1000)

    def fit(self, X, y=None):
        X_transformed = self.vectorizer.fit_transform(X)
        self.lda.fit(X_transformed)
        return self

    def transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        return self.lda.transform(X_transformed)


df = pd.read_csv('/content/startup_ideas (2).csv')


df.dropna(inplace=True)


stop_words = stopwords.words('english')

def preprocess_text(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

df['Startup Idea'] = df['Startup Idea'].apply(preprocess_text)


X = df['Startup Idea']
y = df['Outcome']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tfidf = TfidfVectorizer(max_features=2000)

lda = LdaTransformer(n_components=15, random_state=42)  
gbc = GradientBoostingClassifier(
    n_estimators=300,  
    learning_rate=0.05,  
    max_depth=5
    subsample=0.8,  
    random_state=42
)

smote = SMOTE(random_state=42)

combined_features = FeatureUnion([('tfidf', tfidf), ('lda', lda)])


pipeline = ImbalancedPipeline([
    ('features', combined_features),
    ('smote', smote),
    ('gbc', gbc)
])


pipeline.fit(X_train, y_train)


preds = pipeline.predict(X_test)
pred_probs = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, preds)
print(f"Combined TF-IDF + LDA + Gradient Boosting Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

def predict_success_rate(ideas, model, threshold=0.5):
    preprocessed_ideas = [preprocess_text(idea) for idea in ideas]
    preds = model.predict(preprocessed_ideas)
    probs = model.predict_proba(preprocessed_ideas)[:, 1]  # 
    results = []
    for pred, prob in zip(preds, probs):
        result = "Success" if prob >= threshold else "Failure"
        results.append((result, prob))
    return results

while True:
    user_ideas = []
    while True:
        user_idea = input("Enter your startup idea (<= 20 words) or 'done' to finish: ")
        if user_idea.lower() == 'done':
            if user_ideas:  
                break
        elif len(user_idea.split()) > 20:
            print("Please enter an idea with 20 words or less.")
        else:
            user_ideas.append(user_idea)

    if not user_ideas:
        print("No ideas entered. Exiting.")
        break

    threshold = 0.5  
    predictions = predict_success_rate(user_ideas, pipeline, threshold)

    for i, (idea, (result, prob)) in enumerate(zip(user_ideas, predictions)):
        print(f"Idea {i+1}: '{idea}' -> Prediction: {result}, Success Rate: {prob:.2f}")

    cont = input("Do you want to enter more ideas? (yes/no): ")
    if cont.lower() != 'yes':
        break

this the code what are all the requirements and packages?
