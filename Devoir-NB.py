import pandas as pd
import numpy as np
from gensim.parsing import remove_stopwords, strip_punctuation, strip_short
from sklearn.feature_selection import chi2
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_excel('C:/Users/pc/Desktop/tp3.xlsx')
# print(dataset)
dataset['category_id'] = dataset['category'].factorize()[0]
print(dataset.groupby('category').category_id.count())

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(dataset.text).toarray()
labels = dataset.category_id

category_to_id = {'business': 0, 'tech': 1, 'politics': 2, 'sport': 3, 'entertainment': 4}
id_to_category = {0: 'business', 1: 'tech', 2: 'politics', 3: 'sport', 4: 'entertainment'}

print(features.shape)
print(dataset.shape)
print(dataset.head(2))
target_category = dataset['category'].unique()
print(target_category)
dataset['categoryId'] = dataset['category'].factorize()[0]
print(dataset.head())
category = dataset[["category", "categoryId"]].drop_duplicates().sort_values('categoryId')
print(category)
CategoryCounter = dataset.groupby('category').categoryId.count()
print(CategoryCounter)
text = dataset["text"]
print(text.head())
a = []
for text in dataset:
    filtered_sentence = remove_stopwords(text)
    # print(filtered_sentence)
    res = ''.join([i for i in filtered_sentence if not i.isdigit()])
    new_res = strip_short(res)
    res_new1 = strip_punctuation(new_res)
    p = PorterStemmer()
    p.stem(res_new1)
    # print(res_new1)
    # print('--------------------------------------------------------------------------------------------------')
    a.append(res_new1)

# print(dataset["text"])
text = dataset['text']
category = dataset['category']
text.head()
X_train, X_test, Y_train, Y_test = train_test_split(text, category, test_size=0.3, random_state=60, shuffle=True,
                                                    stratify=category)

print(len(X_train))
print(len(X_test))
nb = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', MultinomialNB()),
               ])
nb.fit(X_train, Y_train)

test_predict = nb.predict(X_test)
train_accuracy = round(nb.score(X_train, Y_train) * 100)
test_accuracy = round(accuracy_score(test_predict, Y_test) * 100)

print("Naive Bayes Train Accuracy Score : {}% ".format(train_accuracy))
print("Naive Bayes Test Accuracy Score  : {}% ".format(test_accuracy))
print()
print(classification_report(test_predict, Y_test, target_names=target_category))
