# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wRjS9t6NvrlvDI9ywDMr2445RaBLOVdR
"""

import os
os.getcwd()

# Commented out IPython magic to ensure Python compatibility.
from nltk import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
# %matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

import nltk
nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

#Mouting the drive to load a simple data set stored on the Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd

train_data = pd.read_csv("/content/gdrive/My Drive/liar_dataset/train.tsv", sep='\t')
test_data = pd.read_csv("/content/gdrive/My Drive/liar_dataset/test.tsv", sep='\t')
valid_data = pd.read_csv("/content/gdrive/My Drive/liar_dataset/valid.tsv", sep='\t')

#Naming the columns of all three datasets

train_data.columns = ['ID',"Label", "Statement", "Subject", "Speaker", "Job Title", "State", "Party", "Barely True Counts", "False Counts", "Half True Counts", "Mostly True Counts", "Pants on Fire Count", "Context"]
test_data.columns = ['ID',"Label", "Statement", "Subject", "Speaker", "Job Title", "State", "Party", "Barely True Counts", "False Counts", "Half True Counts", "Mostly True Counts", "Pants on Fire Count", "Context"]
valid_data.columns = ['ID',"Label", "Statement", "Subject", "Speaker", "Job Title", "State", "Party", "Barely True Counts", "False Counts", "Half True Counts", "Mostly True Counts", "Pants on Fire Count", "Context"]

train_data['Label']=train_data['Label'].apply(lambda x: 1 if x=='false' or x=='barely-true' or x=='pants-fire' else 0)
train_data.head()

test_data['Label']=test_data['Label'].apply(lambda x: 1 if x=='false' or x=='barely-true' or x=='pants-fire' else 0)
test_data.head()

valid_data['Label']=valid_data['Label'].apply(lambda x: 1 if x=='false' or x=='barely-true' or x=='pants-fire' else 0)
valid_data.head()

# Concatenate the two DataFrames vertically
fakeNews_data = pd.concat([train_data, test_data,valid_data ])

fakeNews_data.shape

train_data.head()

#Checking if there are missing values in the dataset
fakeNews_data.isnull().sum()

fakeNews_data = fakeNews_data.dropna()

fakeNews_data.shape

fakeNews_data.drop(columns=['ID',"Barely True Counts", "False Counts", "Half True Counts", "Mostly True Counts", "Pants on Fire Count"], inplace=True)

from nltk.tokenize import word_tokenize

#Tokenization
fakeNews_data['tokenized_statement'] = fakeNews_data['Statement'].apply(lambda x: word_tokenize(x))

#Stopword removal
stop_words = set(stopwords.words('english'))
fakeNews_data['stopword_removal'] = fakeNews_data['tokenized_statement'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

#Lemmatization
lemmatizer = WordNetLemmatizer()
fakeNews_data['lemmatized_statement'] = fakeNews_data['stopword_removal'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

#vectorization

#TF-IDF
tfidf_vectorizer = TfidfVectorizer()
fakeNews_data['lemmatized_statement_str'] = fakeNews_data['lemmatized_statement'].apply(lambda x: ' '.join(x))
tfidf_matrix = tfidf_vectorizer.fit_transform(fakeNews_data['lemmatized_statement_str'])
tfidf_dataset = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

#Bag of Words
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(fakeNews_data['lemmatized_statement_str'])
bow_dataset = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

#N-gram Models
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
ngram_matrix = ngram_vectorizer.fit_transform(fakeNews_data['lemmatized_statement_str'])
ngram_dataset = pd.DataFrame(ngram_matrix.toarray(), columns=ngram_vectorizer.get_feature_names_out())

#Hashing Vectorizer
hashing_vectorizer = HashingVectorizer(n_features=10000)
hashed_matrix = hashing_vectorizer.transform(fakeNews_data['lemmatized_statement_str'])
hashed_dataset = pd.DataFrame(hashed_matrix.toarray())

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

x = [tfidf_dataset, bow_dataset, ngram_dataset, hashed_dataset]
s = ["TFIDF", "Bag of Words", "N-gram", "hashed"]
y = fakeNews_cleaned['Label']
xgb_classifier = XGBClassifier()

for i in range(len(x)):

    spam_training_data,spam_test_data, spam_training_target , spam_test_target = train_test_split(x[i], y, test_size=0.2, random_state=42)
    xgb_classifier.fit(spam_training_data, spam_training_target)
    y_pred = xgb_classifier.predict(spam_test_data)
    accuracy = accuracy_score(spam_test_target_predict, y_pred)
    print("Accuracy:", accuracy)

x = hashed_dataset
y = fakeNews_data['Label']
spam_training_data,spam_test_data, spam_training_target , spam_test_target = train_test_split(x, y, test_size=0.2, random_state=42)

spam_training_data.shape
"""clf = RandomForestClassifier(random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
Accuracy = accuracy_score(y_test,y_pred)
print(Accuracy)"""

spam_test_data.shape

from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

highest_accuracy = 0
penalty_LR = [ "l2",  None]
multi_class_LR = ["auto","ovr", "multinomial"]
max_iter_LR = [ 500]
u = ""
d = " "
f = 0

for g in penalty_LR:
  for h in multi_class_LR:
    for k in max_iter_LR:
      clf_lr = LogisticRegression(penalty = g, multi_class = h, max_iter = k,  random_state = 101)
      clf_lr.fit(spam_training_data,spam_training_target)
      spam_test_target_predict=clf_lr.predict(spam_test_data)
      print("For penalty = ",g,", and multi_class = ", h ,"and max_iter: ",k, "the accuracy score is: ", accuracy_score(spam_test_target,spam_test_target_predict))

      if accuracy_score(spam_test_target,spam_test_target_predict) > highest_accuracy:
          highest_accuracy = accuracy_score(spam_test_target,spam_test_target_predict)
          u = g
          d = h
          f = k

print("The Logistic Regression with the highest accuracy ",highest_accuracy, "has the following parameters: penalty = ", u, " and multi_class = ", d, " and max_iteration = ", f)

clf_lr = LogisticRegression(penalty = u ,multi_class = d , max_iter= f , random_state = 101)
clf_lr.fit(spam_training_data,spam_training_target)
spam_test_target_predict=clf_lr.predict(spam_test_data)
c_m_lr = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r_lr = classification_report(spam_test_target,spam_test_target_predict)
a_s_lr = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",clf_lr.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m_lr)
print(c_r_lr)
print(a_s_lr)

plt.figure(figsize=(5,5))
sns.heatmap(c_m_lr,annot=True,fmt='d', cmap="Greens")

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

clf_dt = DecisionTreeClassifier(criterion = "entropy", max_features = None, splitter = "best", random_state = 101,max_depth = 12 )
clf_gnb = GaussianNB()
eclf = VotingClassifier(estimators = [('DT', clf_dt),('LR', clf_lr), ('GNB', clf_gnb)], voting = 'hard')
eclf.fit(spam_training_data,spam_training_target)
spam_test_target_predict=eclf.predict(spam_test_data)
c_m_VC = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r_VC = classification_report(spam_test_target,spam_test_target_predict)
a_s_VC = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",eclf.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m_VC)
print(c_r_VC)
print(a_s_VC)

plt.figure(figsize=(5,5))
sns.heatmap(c_m_VC,annot=True,fmt='d', cmap="Blues")

ada = AdaBoostClassifier(
           estimator=clf_dt,
           random_state=101)


ada.fit(spam_training_data,spam_training_target)


spam_test_target_predict=ada.predict(spam_test_data)
c_m1 = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r1 = classification_report(spam_test_target,spam_test_target_predict)
a_s1 = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",ada.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m1)
print(c_r1)
print(a_s1)

plt.figure(figsize=(5,5))
sns.heatmap(c_m1,annot=True,fmt='d', cmap="BuPu")

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

criterion_DT = ["gini", "entropy"]
splitter_DT = ["best", "random"]
max_features_DT = ["sqrt", "log2", None]
max_depth_DT = [2,4,6,8,10,12]
highest_accuracy_DT = 0
f = ""
g = ""
h = ""
y = ""
for y in max_depth_DT:
  for x in max_features_DT:
    for n in criterion_DT:
      for z in splitter_DT:
        clf = DecisionTreeClassifier(criterion = n, max_features = x, splitter= z, max_depth = y,random_state = 101)
        clf.fit(spam_training_data,spam_training_target)
        spam_test_target_predict=clf.predict(spam_test_data)
        print("For criterion_DT = ",n,", and max_features_DT = ", x , "and splitter_DT = ", z, "max_depth_DT = ", y, " the accuracy score is: ", accuracy_score(spam_test_target,spam_test_target_predict))

        if accuracy_score(spam_test_target,spam_test_target_predict) > highest_accuracy_DT:
          highest_accuracy_DT = accuracy_score(spam_test_target,spam_test_target_predict)
          f = n
          g = x
          h = z
          w = y

print("The decision tree with the highest accuracy ",highest_accuracy_DT, "has the following parameters: \ncriterion_DT = ", f,"max_depth_DT = ",y , " max_features_DT = ", g, "splitter_DT = ", h)

clf = DecisionTreeClassifier(criterion = f, max_features = g, splitter = h, random_state = 101,max_depth = y )
clf.fit(spam_training_data,spam_training_target)


spam_test_target_predict=clf.predict(spam_test_data)
c_m = confusion_matrix(spam_test_target,spam_test_target_predict)
c_r = classification_report(spam_test_target,spam_test_target_predict)
a_s = accuracy_score(spam_test_target,spam_test_target_predict)


# Compare observed value and Predicted value
print("Prediction for 20 observation:    ",clf.predict(spam_test_data[0:20]))
print("Actual values for 20 observation: ",spam_test_target[0:20].values)
print(c_m)
print(c_r)
print(a_s)

plt.figure(figsize=(5,5))
sns.heatmap(c_m,annot=True,fmt='d', cmap="PuBuGn")

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

x = [tfidf_dataset, bow_dataset, ngram_dataset, hashed_dataset]
s = ["TFIDF", "Bag of Words", "N-gram", "hashed"]
y = fakeNews_data['Label']
xgb_classifier = XGBClassifier()

for i in range(len(x)):

    spam_training_data,spam_test_data, spam_training_target , spam_test_target = train_test_split(x[i], y, test_size=0.2, random_state=42)
    xgb_classifier.fit(spam_training_data, spam_training_target)
    y_pred = xgb_classifier.predict(spam_test_data)
    accuracy = accuracy_score(spam_test_target_predict, y_pred)
    print("Accuracy:", accuracy)