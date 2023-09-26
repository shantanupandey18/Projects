# -*- coding: utf-8 -*-
"""YelpBiz:NLP_&_ML_for_Review_Usefulness_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1roDCShpeKGI3yErfzavYJE9y2IhhPHny

## Part 1: Process and Analyze Yelp Review Dataset (1 point)
The Yelp dataset on user reviews of Tuscon businesses. Specifically, this dataset has six attributes whose descriptions are listed below:
* `review_id`: a string-typed attribute indicating a review's ID
* `user_id`: a string-typed attribute indicating the reviewer's ID
* `business_id`: a string-typed attribute indicating the ID of the business that is reviewed
* `review_stars`: a float-typed attribute indicating the review's star rating
* `useful`: an integer-typed attribute indicating how many useful votes the review has received
* `review_text`: a string-typed attribute storing the review's text
"""

from urllib.request import urlretrieve

urlretrieve('https://drive.google.com/uc?export=download&id=11okLSqOwXBgOLw2lKXb0Bi_yMzoeC5u1',
            'user_reviews.csv')

# Read csv
import pandas as pd
data = pd.read_csv('user_reviews.csv', sep=',')

# Print dataframes
print(data)

# Remove duplicates & null for business id and review id
data = data.dropna(axis = "index", how = "any").drop_duplicates(subset=["review_id", "business_id"], keep="last");

# list top 20 users with most reviews
df_metrics = data.groupby(['user_id']).agg(review_count = ('review_id', 'count'),
                                           useful_count = ('useful', 'count'),
                                           avg_stars = ('review_stars', 'mean')).sort_values(by = 'review_count',
                                                                                           ascending = False).head(20)

print(df_metrics)

# List top 20 businesses
df_business_metrics = data.groupby(['business_id']).agg(review_count = ('review_id', 'count'),
                                                        avg_stars = ('review_stars', 'mean')).sort_values(by = 'review_count',
                                                                                           ascending = False).head(20)
print(df_business_metrics)

"""## Part 2: Preprocess Review Text
 We first tokenize each review text, lowercase each token, remove punctuations and stop words, and conduct stemming for each remaining token. After finishing the preprocessing, we combine all remaining tokens of a review back to a single string. Print the first 25 preprocessed reviews.
"""

## download the punkt module
import nltk
nltk.download('punkt')

## get punctuations
import string
punctuations = string.punctuation

## download stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

## import stemmer
from nltk.stem import PorterStemmer
ps = PorterStemmer()

usefulness = data[['review_text', 'useful']]

new_train_msgs = []
review_tokenized_list = []
review_lowercased = []
review_tokenized_lowercase_list = []
print('Result for nltk tokenization:')
for msg in usefulness.review_text:
    ## TODO: insert your code here to process each message in the spam-train.csv
    review_tokenized_list.append(nltk.word_tokenize(msg))
    review_lowercased.append(msg.lower())


for labels in usefulness.review_text:
    review_tokenized_lowercase_list.append(nltk.word_tokenize(labels))


review_no_punctuations_words = []
review_no_punctuations_list = []
for words in review_tokenized_lowercase_list:
    for word in words:
        if word not in string.punctuation:
            review_no_punctuations_words.append(word)
    review_no_punctuations_list.append(review_no_punctuations_words)
    review_no_punctuations_words = []

# remove stop words
stop_words = set(stopwords.words('english'))
review_no_stopwords = []
review_no_stopwords_list = []
for label in review_no_punctuations_list:
    for word in label:
      if word not in stop_words:
          review_no_stopwords.append(word)
    review_no_stopwords_list.append(review_no_stopwords)
    review_no_stopwords = []

# stemming the text
review_stemmed = []
review_stemmed_list_train = []
for token in review_no_stopwords_list:
  for word in token:
    review_stemmed.append(ps.stem(word))
  review_stemmed_list_train.append(review_stemmed)
  review_stemmed = []

### append the processed review into new list
final_string_train = []
for strings in review_stemmed_list_train:
    review_processed_string = " ".join(strings)
    final_string_train.append(review_processed_string)
    review_processed_string = ""

# usefulness.review_text = final_string_train
data.review_text = final_string_train

print(data.review_text)

"""## Part 3: Predict Review Usefulness based on Their Text
Finally, in this part, we select 2 appropriate machine learning models to predict each review's usefulness (i.e., number of useful votes) based on its text and report the models' prediction performance, namely KNN and Naive Bayes




"""

from sklearn.model_selection import train_test_split

x = data['review_text']
y = data['useful']
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

# we can apply TfidfVectorizer function to extract TF-IDF vectors for each message
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_transformer = TfidfVectorizer(max_features=500)

# TODO: replace the question marks '?' with the correct variables you created in Section 2.1
tfidf_transformer.fit(x)
train_features = tfidf_transformer.transform(x_train).toarray()
test_features = tfidf_transformer.transform(x_test).toarray()

print(train_features)
print(test_features)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,\
    precision_score, f1_score, roc_auc_score

# Training the dataset
knn = KNeighborsClassifier(n_neighbors=5)           # K = 5
knn.fit(train_features, y_train)                                       # fit the model
knn_pred_t = knn.predict(test_features)                       # make predictions
knn_score_t = knn.predict_proba(test_features)                # get prediction scores

## print the predicted labels
print('Predicted labels testing:')
print(knn_pred_t)
print()

## print the prediction scores
print('Predicted scores testing:')
print(knn_score_t)
print()


# calculate prediction performance
print('Confusion Matrix testing:')
knn_conf_mat = confusion_matrix(y_test, knn_pred_t)
print(knn_conf_mat)
print()

## accuracy
knn_acc = accuracy_score(y_test, knn_pred_t)
print('Prediction accuracy : {:.4f}'.format(knn_acc))

## precision
knn_precision = precision_score(y_test, knn_pred_t,average = "weighted")
print('Prediction precision : {:.4f}'.format(knn_precision))

## F1 score
knn_f1 = f1_score(y_test, knn_pred_t,average = "weighted")
print('Prediction F1 : {:.4f}'.format(knn_f1))

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,\
    precision_score, f1_score, roc_auc_score

# Model ran
gnb = GaussianNB()
gnb.fit(train_features, y_train)                  # fit the model
gnb_pred_t = gnb.predict(test_features)                       # make predictions
gnb_score_t = gnb.predict_proba(test_features)                # get prediction scores

## accuracy
gnb_acc = accuracy_score(y_test, gnb_pred_t)
print('Prediction accuracy: {:.4f}'.format(gnb_acc))

## recall
gnb_recall = recall_score(y_test, gnb_pred_t,average = "weighted")
print('Prediction recall: {:.4f}'.format(gnb_recall))

## precision
gnb_precision = precision_score(y_test, gnb_pred_t,average = "weighted")
print('Prediction precision: {:.4f}'.format(gnb_precision))

## F1 score
gnb_f1 = f1_score(y_test, gnb_pred_t,average = "weighted")
print('Prediction F1: {:.4f}'.format(gnb_f1))