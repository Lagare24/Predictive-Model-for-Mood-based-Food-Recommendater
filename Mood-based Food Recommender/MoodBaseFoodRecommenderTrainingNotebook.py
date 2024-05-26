#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn import metrics


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[108]:


df = pd.read_csv("food_choices.csv")
selected_columns = ['comfort_food', 'comfort_food_reasons', 'comfort_food_reasons_coded', 'Gender']
df = df[selected_columns]


# In[109]:


df['comfort_food'].fillna('unknown', inplace=True)
df['comfort_food_reasons'].fillna('unknown', inplace=True)
mode_coded = df['comfort_food_reasons_coded'].mode()[0]
df['comfort_food_reasons_coded'].fillna(mode_coded, inplace=True)

missing_values = df.isnull().sum()
print("Missing values after handling:\n", missing_values)


# In[110]:


def categorize_food(food):
    food = food.lower()
    if any(keyword in food for keyword in ['fruit', 'vegetable', 'grapes', 'carrots', 'broccoli', 'tomato soup', 'spaghetti squash', 'carrots', 'plantain chips', 'almonds', 'watermelon', 'cucumber', 'fritos']):
        return 'Fruit and Vegetables'
    elif any(keyword in food for keyword in ['pizza', 'pasta', 'spaghetti', 'noodles', 'rice', 'potatoes', 'bagels', 'fries', 'taco', 'sandwich', 'hot dog', 'burrito', 'sub', 'macaroni and cheese', 'lasagna', 'mashed potatoes', 'spaghetti squash', 'stuffed peppers', 'meatball sub', 'chicken tikka masala', 'chicken noodle soup', 'chicken pot pie']):
        return 'Starchy food'
    elif any(keyword in food for keyword in ['milk', 'cheese', 'yogurt', 'ice cream', 'mozzarella sticks', 'cottage cheese', 'cheesecake', 'frozen yogurt']):
        return 'Dairy'
    elif any(keyword in food for keyword in ['chicken', 'beef', 'pork', 'fish', 'egg', 'turkey', 'meatball', 'sausage', 'fried chicken', 'grilled chicken', 'chicken fingers', 'beef jerky', 'steak']):
        return 'Protein'
    elif any(keyword in food for keyword in ['chips', 'fries', 'chocolate', 'cake', 'cookie', 'brownie', 'candy', 'soda', 'donut', 'peanut butter', 'burgers', 'garlic bread', 'popcorn', 'pretzels', 'chicken wings', 'doughnut', 'chocolate bar', 'twizzlers', 'chocolate brownie', 'macaroons', 'truffles', 'french fries', 'slim jims', 'chicken curry', 'chocolate chipotle', 'pop', 'mac n cheese', 'rice', 'pizza', 'cheeseburger', 'chicken nuggets', 'peanut butter sandwich', 'mac and cheese', 'cheese and crackers', 'protein bar', 'chex mix', 'cheez-its', 'chicken fingers', 'chips and cheese', 'chips and dip', 'fruit snacks', 'doritos']):
        return 'Fat'
    else:
        return 'Others'


# In[111]:


df['food_group'] = df['comfort_food'].apply(categorize_food)


# In[112]:


def map_reasons_to_mood(reasons):
    reason_to_mood = {
        'stress': 'stress/anxiety',
        'anxiety': 'stress/anxiety',
        'boredom': 'boredom',
        'sadness': 'sadness',
        'happiness': 'happiness/celebration',
        'celebration': 'happiness/celebration',
        'other': 'other',
        'none': 'other',
        'unknown': 'other'
    }
    
    reasons_list = [r.strip().lower() for r in reasons.split(',')]
    
    mood_categories = [reason_to_mood.get(reason, 'other') for reason in reasons_list]
    
    return mood_categories


# In[113]:


df['mood_category'] = df['comfort_food_reasons'].apply(map_reasons_to_mood)


# In[114]:


ohe = OneHotEncoder()
encoded_features = ohe.fit_transform(df[['Gender', 'food_group', 'comfort_food_reasons_coded']]).toarray()


# In[115]:


X = encoded_features
y = df['mood_category'].apply(lambda x: x[0])


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[119]:


param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

param_grid_lr = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}

classifiers = {
    "LogisticRegression": GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5),
    "GradientBoosting": GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5)
}


# In[120]:


for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f"Classification Report for {name}:\n", report)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy}\n")

    model_filename = f"{name}_model.pkl"
    joblib.dump(clf, model_filename)
    print(f"Saved {name} model to {model_filename}")


# In[121]:


preprocessed_file = "preprocessed_data.csv"
df.to_csv(preprocessed_file, index=False)
print(f"Preprocessed data saved to {preprocessed_file}")


# In[122]:


training_artifacts = {
    "preprocessed_data_file": preprocessed_file,
    "one_hot_encoder": ohe,
    "tfidf_vectorizer": vectorizer,
    "models": classifiers
}

joblib.dump(training_artifacts, "training_artifacts.joblib")
print("Training artifacts saved")


# In[123]:


joblib.dump(ohe, 'one_hot_encoder.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Saved preprocessors to one_hot_encoder.pkl and tfidf_vectorizer.pkl")


# In[124]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['comfort_food_reasons']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Comfort Food Reasons')
plt.axis('off')
plt.show()


# In[125]:


df_exploded = df.explode('mood_category')

plt.figure(figsize=(8, 6))
sns.countplot(y='mood_category', data=df_exploded, order=df_exploded['mood_category'].value_counts().index, palette='viridis')
plt.title('Distribution of Mood Categories')
plt.xlabel('Count')
plt.ylabel('Mood Category')
plt.show()


# In[126]:


gender_mapping = {1: 'Female', 2: 'Male'}
df['Gender'] = df['Gender'].map(gender_mapping)

plt.figure(figsize=(6, 6))
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'skyblue'], explode=(0.1, 0), startangle=140)
plt.title('Gender Distribution')
plt.ylabel('')
plt.axis('equal')
plt.show()


# In[128]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='comfort_food_reasons_coded', data=df, palette='Set2')
plt.title('Comfort Food Frequency by Gender')
plt.xlabel('Gender')
plt.ylabel('Comfort Food Frequency (Coded)')
plt.show()


# In[129]:


df_exploded = df.explode('mood_category')

correlation_matrix = df_exploded.groupby(['food_group', 'mood_category']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='d', linewidths=0.5)
plt.title('Correlation between Food Groups and Mood Categories')
plt.xlabel('Mood Category')
plt.ylabel('Food Group')
plt.show()


# In[130]:


for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-Validation Scores for {name}: {scores}")
    print(f"Mean Cross-Validation Score for {name}: {scores.mean()}\n")


# In[ ]:




