import pandas as pd
import numpy as np
# import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer
import string
import re
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.like = None
        self.priors = None
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount

    def fit(self, x, yy):
        nb_examples = len(x)

        self.priors = np.bincount(yy) / nb_examples
        print('Priors:')
        print(self.priors)

        occs = np.zeros((self.nb_classes, self.nb_words))
        for i in range(nb_examples):
            c = yy[i]
            occs[c] += x[i]
        print('Occurrences:')
        print(occs)

        self.like = (occs + self.pseudocount) / (np.sum(occs, axis=1)[:, np.newaxis] + self.nb_words * self.pseudocount)
        print('Likelihoods:')
        print(self.like)

    def predict(self, x, class_names=None):
        log_probs = np.log(self.priors) + x @ np.log(self.like.T)
        predictions = np.argmax(log_probs, axis=1)
        '''
        if class_names is not None:
            predicted_classes = [class_names[idx] for idx in predictions]
            for i, class_name in enumerate(predicted_classes):
                print(f"Document {i + 1}: Predicted class - {class_name}")

        else:
            for i, idx in enumerate(predictions):
                print(f"Document {i + 1}: Predicted class index - {idx}")
        '''
        return predictions


def preprocess_tweets(tweet):
    tweet = re.sub(r'&amp;', '', tweet)
    tweet = re.sub(r'&gt;', '', tweet)
    tweet = re.sub(r'&lt;', '', tweet)
    tweet = re.sub(r"'", '', tweet)
    tweet = re.sub(r"'s", '', tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r"[.']", '', tweet)
    tokens = word_tokenize(tweet)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation) | {'.'}
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(token) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [re.sub(r'(.)\1+', r'\1', token) for token in tokens]
    return ' '.join(tokens)


def create_feature_vectors(tweets, vocabulary):
    vectors = np.zeros((len(tweets), len(vocabulary)))
    for i, tweet in enumerate(tqdm(tweets, desc="Creating feature vectors")):
        word_counts = Counter(tweet.split())
        for j, word in enumerate(vocabulary):
            vectors[i][j] = word_counts[word]
    return vectors


df = pd.read_csv('disaster-tweets.csv')
df['clean_tweets'] = df['text'].apply(preprocess_tweets)

all_words = ' '.join(df['clean_tweets']).split()
vocab = list(set(all_words))[:10000]

X = create_feature_vectors(df['clean_tweets'], vocab)
y = df['target'].values

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

start_time = time.time()

model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)
model.fit(X_train, y_train)

end_time = time.time()
print("Training completed in {:.2f} seconds".format(end_time - start_time))

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.figure(figsize=(6, 4))
df['target'].value_counts().plot(kind='bar')

categories = ['Non-Disaster', 'Disaster']
actual_counts_test = np.bincount(y_test)
predicted_counts = np.bincount(y_pred, minlength=len(actual_counts_test))

bar_width = 0.35
actual_positions = np.arange(len(categories))
predicted_positions = actual_positions + bar_width

plt.figure(figsize=(8, 5))
plt.bar(actual_positions, actual_counts_test, width=bar_width, color='green', label='Actual (Test Set)')
plt.bar(predicted_positions, predicted_counts, width=bar_width, color='red', label='Predicted')

plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Comparison of Actual and Predicted Labels in Test Set')
plt.xticks(actual_positions + bar_width / 2, categories)
plt.legend()

plt.show()

positive_tweets = df[df['target'] == 1]['clean_tweets']
negative_tweets = df[df['target'] == 0]['clean_tweets']

positive_word_counts = Counter(' '.join(positive_tweets).split())
negative_word_counts = Counter(' '.join(negative_tweets).split())

top_positive_words = positive_word_counts.most_common(5)
top_negative_words = negative_word_counts.most_common(5)
print("Top 5 words in positive tweets:", top_positive_words)
print("Top 5 words in negative tweets:", top_negative_words)

common_words = set(word for word in positive_word_counts if positive_word_counts[word] >= 10 and negative_word_counts[word] >= 10)
lr_metric = {word: positive_word_counts[word] / negative_word_counts[word] for word in common_words}

top_lr_words = sorted(lr_metric.items(), key=lambda x: x[1], reverse=True)[:5]
bottom_lr_words = sorted(lr_metric.items(), key=lambda x: x[1])[:5]
print("Top 5 words by LR metric:", top_lr_words)
print("Bottom 5 words by LR metric:", bottom_lr_words)

# print(df.sample(10)['clean_tweets'].values)
