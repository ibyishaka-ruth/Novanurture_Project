import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# STEP 1: DATA INTEGRATION
# =====================================================

print("\nSTEP 1: DATA INTEGRATION\n")

data1 = {
    "text": [
        "I am really sad and lonely.",
        "Everything makes me angry lately.",
        "I’m anxious about my exams.",
        "Life feels amazing and beautiful!"
    ],
    "emotion": ["sadness", "anger", "fear", "joy"]
}

df1 = pd.DataFrame(data1)

data2 = {
    "text": ["I feel so happy today!", "I am scared about tomorrow."],
    "emotion": ["joy", "fear"]
}

df2 = pd.DataFrame(data2)

df_combined = pd.concat([df1, df2], ignore_index=True)

# Add extra columns
df_combined["user_id"] = [101, 102, 103, 104, 105, 106][:len(df_combined)]
df_combined["timestamp"] = pd.date_range("2025-01-01", periods=len(df_combined), freq="3H")

print("AFTER INTEGRATION:")
print(df_combined.head(), "\n")

# =====================================================
# STEP 2: DATA CLEANING
# =====================================================

print("\nSTEP 2: DATA CLEANING\n")

before_rows = len(df_combined)
df_combined.drop_duplicates(inplace=True)
df_combined.dropna(inplace=True)
after_rows = len(df_combined)

print(f"Rows before cleaning: {before_rows}")
print(f"Rows after cleaning: {after_rows}\n")

# =====================================================
# STEP 3: DATA REDUCTION
# =====================================================

print("\nSTEP 3: DATA REDUCTION\n")

columns_to_keep = ["text", "emotion", "timestamp"]
df_reduced = df_combined[columns_to_keep]

print(f"All columns before reduction: {list(df_combined.columns)}")
print(f"Columns removed during reduction: {[col for col in df_combined.columns if col not in columns_to_keep]}\n")

print("BEFORE REDUCTION:")
print(df_combined.head(5))
print("\nAFTER REDUCTION:")
print(df_reduced.head(5))

# =====================================================
# STEP 4: DATA TRANSFORMATION
# =====================================================

print("\nSTEP 4: DATA TRANSFORMATION\n")

# Convert text to lowercase
df_reduced["text"] = df_reduced["text"].str.lower()

# Define stopwords manually
stop_words = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","do","does","did","a",
    "an","the","and","but","if","or","because","as","until","while","of","at",
    "by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too","very"
])

# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# Apply text cleaning
df_reduced["clean_text"] = df_reduced["text"].apply(remove_stopwords)

# Label Encoding for emotions
le = LabelEncoder()
df_reduced["emotion_label"] = le.fit_transform(df_reduced["emotion"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_reduced["clean_text"])

# Display before & after transformation
print("BEFORE TRANSFORMATION:")
print(df_reduced[["text", "emotion"]].head(), "\n")

print("AFTER TRANSFORMATION (cleaned + encoded):")
print(df_reduced[["clean_text", "emotion_label"]].head(), "\n")

print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}\n")

# Compare word counts before & after
df_reduced["word_count_before"] = df_reduced["text"].apply(lambda x: len(x.split()))
df_reduced["word_count_after"] = df_reduced["clean_text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(5, 3))
plt.bar(["Before", "After"],
        [df_reduced["word_count_before"].mean(), df_reduced["word_count_after"].mean()],
        color=['red', 'green'])
plt.title("Effect of Data Transformation (Avg Word Count)")
plt.ylabel("Average Words per Sentence")
plt.tight_layout()
plt.show()

# =====================================================
# STEP 5: DATA DISCRETIZATION
# =====================================================

print("\nSTEP 5: DATA DISCRETIZATION\n")

# Function to categorize timestamp into time of day
def time_to_period(time):
    hour = time.hour
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"

df_reduced["time_period"] = df_reduced["timestamp"].apply(time_to_period)

print("BEFORE DISCRETIZATION:")
print(df_reduced[["timestamp"]].head(), "\n")

print("AFTER DISCRETIZATION:")
print(df_reduced[["timestamp", "time_period"]].head(), "\n")

# Count records by time period
time_counts = df_reduced["time_period"].value_counts()

plt.figure(figsize=(5,3))
plt.bar(time_counts.index, time_counts.values, color='skyblue')
plt.title("Effect of Data Discretization (Time Categories)")
plt.ylabel("Number of Records")
plt.xlabel("Time Period")
plt.tight_layout()
plt.show()

print("\nDiscretization complete — reflections grouped into Morning, Afternoon, Evening, or Night.\n")
