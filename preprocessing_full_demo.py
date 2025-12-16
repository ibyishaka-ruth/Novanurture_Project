# preprocessing_full_demo_graphs.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1️⃣ DATA GENERATION & INTEGRATION
# -----------------------------
texts_main = [
    "I feel so happy today!",
    "I am really sad and lonely.",
    "Everything makes me angry lately.",
    "I’m anxious about my exams.",
    "Life feels amazing and beautiful!"
]

emotions_main = ["joy", "sadness", "anger", "fear", "joy"]

data_main = {
    "text": [t + f" ({i})" for i in range(10) for t in texts_main],
    "emotion": emotions_main * 10
}

df_main = pd.DataFrame(data_main)

texts_extra = ["I am feeling excited", "I feel a bit angry today", "Life is amazing", "I am scared of the dark"]
emotions_extra = ["joy", "anger", "joy", "fear"]
df_extra = pd.DataFrame({"text": texts_extra*10, "emotion": emotions_extra*10})

df_combined = pd.concat([df_main, df_extra], ignore_index=True)
df_combined["user_id"] = np.arange(101, 101 + len(df_combined))
df_combined["timestamp"] = pd.date_range(start="2025-01-01 06:30", periods=len(df_combined), freq="H")

# -----------------------------
# 2️⃣ DATA CLEANING
# -----------------------------
df_combined.drop_duplicates(inplace=True)
df_combined.dropna(inplace=True)

# -----------------------------
# 3️⃣ DATA REDUCTION
# -----------------------------
columns_to_keep = ["text", "emotion", "timestamp"]
df_reduced = df_combined[columns_to_keep]

# -----------------------------
# 4️⃣ DATA TRANSFORMATION
# -----------------------------

# -----------------------------
stop_words = set(["i","am","so","a","the","and","is","are","of","my","to"])
df_reduced["text"] = df_reduced["text"].str.lower()
df_reduced["clean_text"] = df_reduced["text"].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))

le = LabelEncoder()
df_reduced["emotion_label"] = le.fit_transform(df_reduced["emotion"])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_reduced["clean_text"])

# -----------------------------
# Graph: Effect of Transformation (avg word count before vs after)
# -----------------------------
df_reduced["words_before"] = df_reduced["text"].apply(lambda x: len(x.split()))
df_reduced["words_after"] = df_reduced["clean_text"].apply(lambda x: len(x.split()))

# Graph: Effect of Transformation (avg word count before vs after)
plt.figure(figsize=(5,3))
plt.bar(
    ["Before Transformation", "After Transformation"], 
    [df_reduced["words_before"].mean(), df_reduced["words_after"].mean()],
    color=['blue','lime']   # changed colors to blue and lime
)
plt.title("Effect of Transformation (Avg Words per Sentence)")
plt.ylabel("Average Words")
plt.tight_layout()
plt.savefig("graph_transformation.png")
plt.close()

# -----------------------------
# 5️⃣ DATA DISCRETIZATION
# -----------------------------
def time_to_period(time):
    h = time.hour
    if 6 <= h < 12: return "Morning"
    elif 12 <= h < 18: return "Afternoon"
    elif 18 <= h < 24: return "Evening"
    else: return "Night"

df_reduced["time_period"] = df_reduced["timestamp"].apply(time_to_period)

plt.figure(figsize=(5,3))
df_reduced["time_period"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Reflections by Time Period")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graph_discretization.png")
plt.close()

# -----------------------------
# 6️⃣ DATA AUGMENTATION
# -----------------------------
synonyms = {"happy":"joyful","sad":"unhappy","angry":"mad","scared":"afraid","excited":"thrilled","amazing":"wonderful"}

def augment(text):
    words = text.split()
    new_words = [synonyms[w] if w in synonyms and random.random()>0.5 else w for w in words]
    random.shuffle(new_words)
    return " ".join(new_words)

df_reduced["augmented_text"] = df_reduced["clean_text"].apply(augment)

# Plot example of original vs augmented text lengths
df_reduced["aug_len"] = df_reduced["augmented_text"].apply(lambda x: len(x.split()))
plt.figure(figsize=(5,3))
plt.bar(["Clean Text Avg Length","Augmented Text Avg Length"], 
        [df_reduced["words_after"].mean(), df_reduced["aug_len"].mean()],
        color=['orange','purple'])
plt.title("Effect of Augmentation on Avg Words")
plt.ylabel("Avg Words per Sentence")
plt.tight_layout()
plt.savefig("graph_augmentation.png")
plt.close()

# -----------------------------
# Save final dataset
# -----------------------------
df_reduced.to_csv("final_preprocessed.csv", index=False)
print("All preprocessing steps complete! Graphs saved as:")
print(" - graph_cleaning.png")
print(" - graph_discretization.png")
print(" - graph_augmentation.png")
print("💾 Final dataset saved as 'final_preprocessed.csv'")
