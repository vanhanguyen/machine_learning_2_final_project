#import libraries
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from textblob import TextBlob
from wordcloud import WordCloud
from scipy.stats import ttest_ind
from gensim import corpora, models

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

#Load dataframe
df = pd.read_csv('/Users/imadulislamchowdhury/Downloads/business_intelligence/pinterest.csv')

# Exploratory Data Analysis (EDA)
# Display a concise summary of the DataFrame, including the number of non-null entries and data types of each column
print(df.info())
# Display the first 5 rows of the DataFrame to get a quick overview of the data
print(df.head())
# Generate descriptive statistics that summarize the central tendency, dispersion, and shape of the dataset’s distribution, excluding NaN values
print(df.describe())
# Display the data types of each column in the DataFrame
print(df.dtypes)
# Display the dimensions of the DataFrame (number of rows, number of columns)
print(df.shape)

#visualization
# Extract the 'repin_count' column
repin_count = df['repin_count']

# Create a histogram for the 'repin_count' column
plt.figure(figsize=(10, 6))
plt.hist(repin_count, bins=20, edgecolor='black')
plt.title('Histogram of Repin Count')
plt.xlabel('Repin Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Create a box plot for the 'repin_count' column
plt.figure(figsize=(10, 6))
plt.boxplot(repin_count, vert=False)
plt.title('Box Plot of Repin Count')
plt.xlabel('Repin Count')
plt.grid(True)
plt.show()

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Check the percentage of missing values per column
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values per column:\n", missing_percentage)

df = df.dropna(subset=['description'])
df = df.drop(columns=['title'], errors='ignore')
print(df.shape)

#Sentiment Analysis
# Function to classify sentiment
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['sentiment'] = df['description'].apply(get_sentiment)

# Count the sentiment distribution
sentiment_counts = df['sentiment'].value_counts()

# Print sentiment counts
print(sentiment_counts)

# Plot the sentiment distribution
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Prepare data for model training
X = df['description']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model using cross-validation
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#Keyword Extraction
# Preprocess text: Convert to lowercase, remove NaN, and drop empty descriptions
df['description'] = df['description'].astype(str).str.lower().fillna("")
df = df[df['description'].str.strip() != ""]

#Remove stopwords
stop_words = set(stopwords.words("english"))
df['cleaned_description'] = df['description'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

#TF-IDF Keyword Extraction
vectorizer = TfidfVectorizer(max_features=50)  # Extract top 50 keywords
tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])
keywords = vectorizer.get_feature_names_out()
print("Top Keywords from TF-IDF:\n", keywords)

#LDA Topic Modeling
texts = [text.split() for text in df['cleaned_description']]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
print("\n Key Topics Identified by LDA:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx+1}: {topic}")

#Generate Word Cloud for Visualization
text_combined = " ".join(df['cleaned_description'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_combined)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Pinterest Descriptions")
plt.show()

# Prepare data for topic modeling using CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_description'])

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Train an LDA model for topic modeling
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X_train)

# Perform cross-validation for topic modeling (using perplexity as a metric)
perplexity_scores = []
for i in range(5):
    lda_model.fit(X_train)
    perplexity_scores.append(lda_model.perplexity(X_test))

print("Perplexity Scores:", perplexity_scores)
print("Mean Perplexity Score:", sum(perplexity_scores) / len(perplexity_scores))

#Trend Analysis
# Extract the 'repin_count' column
repin_count = df['repin_count']

# Plot the trend of repin_count over time
plt.figure(figsize=(10, 6))
plt.plot(repin_count)
plt.title('Trend of Repin Count Over Time')
plt.xlabel('Index')
plt.ylabel('Repin Count')
plt.grid(True)
plt.show()

# Calculate and print basic statistics for repin_count
print("Basic Statistics for Repin Count:")
print(repin_count.describe())

# Simulate A/B testing by splitting data into two groups
group_a = df.sample(frac=0.5, random_state=42)
group_b = df.drop(group_a.index)

# Calculate mean repin_count for both groups
mean_repin_a = group_a['repin_count'].mean()
mean_repin_b = group_b['repin_count'].mean()

print("Mean Repin Count for Group A:", mean_repin_a)
print("Mean Repin Count for Group B:", mean_repin_b)

# Perform a t-test to compare the means of the two groups
t_stat, p_value = ttest_ind(group_a['repin_count'], group_b['repin_count'])
print("T-Statistic:", t_stat)
print("P-Value:", p_value)
