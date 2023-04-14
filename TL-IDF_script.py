import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge
import numpy as np

# load the dataset
df = pd.read_csv("news.csv")

# split the dataset into train and test sets
train_size = int(0.9 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# define a function to preprocess the text data
def preprocess(text):
    # convert text to lowercase
    text = text.lower()
    # tokenize text into words
    words = word_tokenize(text)
    # remove stopwords and punctuation
    words = [word for word in words if word not in stopwords.words("english") and word.isalpha()]
    # join words to form sentences
    return " ".join(words)

# preprocess the text data in the test set
test_df["cleaned_content"] = test_df["content"].apply(preprocess)

# compute the TF-IDF score for each sentence in the test set
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5, use_idf=True)
X = vectorizer.fit_transform(test_df["cleaned_content"])
features = vectorizer.get_feature_names_out()
tfidf_scores = []
for i in range(X.shape[0]):
    feature_index = X[i,:].nonzero()[1]
    tfidf_scores.append(dict(zip([features[i] for i in feature_index], [X[i, x] for x in feature_index])))

# rank the sentences based on their importance
sent_scores = []
for scores in tfidf_scores:
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    sent_scores.append(list(scores.keys()))

# select the top 30% of sentences as the summary for each article
summaries = []
for scores in sent_scores:
    num_sent = int(np.ceil(len(scores) * 0.3))
    summary = " ".join(scores[:num_sent])
    summaries.append(summary)

# remove the selected sentences from the original article to generate the cleaned response
cleaned_responses = []
for i in range(len(summaries)):
    summary_words = summaries[i].split()
    content_words = test_df["cleaned_content"].iloc[i].split()
    cleaned_words = [word for word in content_words if word not in summary_words]
    cleaned_response = " ".join(cleaned_words)
    cleaned_responses.append(cleaned_response)

# calculate ROUGE scores
rouge = Rouge()
rouge_scores = rouge.get_scores(pd.Series(cleaned_responses), test_df["cleaned_content"], avg=True)

# print the cleaned responses for the test set along with ROUGE scores
for i in range(len(cleaned_responses)):
    print(f"Original Content: {test_df['content'].iloc[i]}")
    print(f"New Content: {cleaned_responses[i]}")
    print(f"Removed Lines: {summaries[i]}")
    print(f"ROUGE Scores: {rouge_scores}")
    print("\n")
