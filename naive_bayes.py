import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

twenty_train = fetch_20newsgroups(subset = 'train', shuffle = True)
print(twenty_train.target_names)

FULL_LEN = 20000 # Total number of articles
train_len = len(twenty_train.data)
print("%%age of dataset used for training: %3.4f" % (train_len / FULL_LEN * 100.0))

cv = CountVectorizer()
X_train_counts = cv.fit_transform(twenty_train.data)

print("# of articles: %d" % X_train_counts.shape[0])
print("Count for words in the first article: ", repr(X_train_counts[0].data))
print("Indexes of words in the first article: ", repr(X_train_counts[0].indices))
print("The first 5 words in the first article:\n")
for ind in X_train_counts[0].indices[:5]:
   print(cv.get_feature_names_out()[ind]," ", end="")

# raw word counts
clf = MultinomialNB()
clf.fit(X_train_counts, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle = True)
X_test_counts = cv.transform(twenty_test.data)
predicted = clf.predict(X_test_counts)
perf = np.mean(predicted == twenty_test.target)
print("Our 20 Newsgroup's raw-count classifer correctly classified %3.4f%% of the articles." 
      % (perf * 100.0))

# fix raw word counts that occur frequently across documents
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
tfidf_clf = MultinomialNB()
tfidf_clf.fit(X_train_tfidf, twenty_train.target)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
predicted_tfidf = tfidf_clf.predict(X_test_tfidf)
perf_tfidf = np.mean(predicted_tfidf == twenty_test.target)
print("Our 20 Newsgroup's tf-idf classifer correctly classified %3.4f%% of the articles." 
      % (perf_tfidf * 100.0))

# stop words
sw_count_vect = CountVectorizer(stop_words = 'english')
X_train_counts = sw_count_vect.fit_transform(twenty_train.data)
X_test_counts = sw_count_vect.transform(twenty_test.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
tfidf_clf = MultinomialNB()
tfidf_clf.fit(X_train_tfidf, twenty_train.target)
predicted_tfidf_sw = tfidf_clf.predict(X_test_tfidf)
perf_tfidf_sw = np.mean(predicted_tfidf_sw == twenty_test.target)
print("Our 20 Newsgroup's tf-idf with stop-words classifer correctly classified %3.4f%% of the articles." 
      % (perf_tfidf_sw * 100.0))

# use pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()), ])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted_pipeline = text_clf.predict(twenty_test.data)
perf_pipeline = np.mean(predicted_pipeline == twenty_test.target)
print("Our 20 Newsgroup's pipeline classifer correctly classified %3.4f%% of the articles." 
      % (perf_pipeline * 100.0))

# print(text_clf['clf'])