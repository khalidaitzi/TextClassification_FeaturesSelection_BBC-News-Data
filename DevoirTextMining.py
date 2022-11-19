import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn import svm

random_state = 321

data = pd.read_excel('C:/Users/pc/Desktop/tp3.xlsx')
data['category_id'] = data['category'].factorize()[0]
df = pd.DataFrame(list(zip(data['text'], data['category_id'])), columns=['text', 'label'])
print(df.head())

labels, counts = np.unique(df['label'], return_counts=True)
print("Nombre d'articles pour chaque catégorie ")
print('-----------------------------------------------------')
print(dict(zip(data.category, counts)))
print('-----------------------------------------------------')
X_train, X_test, y_train, y_test = train_test_split(df["text"],
                                                    df["label"],
                                                    test_size=0.3,
                                                    random_state=random_state)

tokenize = RegexpTokenizer(r'[a-zA-Z0-9]+')

vectorizer = CountVectorizer(lowercase=True,
                             tokenizer=tokenize.tokenize,
                             stop_words='english',
                             ngram_range=(1, 2),
                             analyzer='word',
                             min_df=3,
                             max_features=None)

News = vectorizer.fit_transform(X_train)
print("Résultat après la vectorisation")
print('-----------------------------------------------------')
print(News.shape)
print('-----------------------------------------------------')

importance = np.argsort(np.asarray(News.sum(axis=0)).ravel())[::-1]
feature_names = np.array(vectorizer.get_feature_names_out())
print(feature_names[importance[:20]])

X_test_vector = vectorizer.transform(X_test)
ch2 = SelectKBest(chi2, k=20)
ch2.fit_transform(News, y_train)
feature_names_chi2 = [feature_names[i] for i
                      in ch2.get_support(indices=True)]
print("Chi2 résultat")
print('-----------------------------------------------------')
print(feature_names_chi2)
print('-----------------------------------------------------')

mutual_info = SelectKBest(mutual_info_classif, k=20)
mutual_info.fit_transform(News, y_train)

feature_names_mutual_info = [feature_names[i] for i
                             in mutual_info.get_support(indices=True)]
print("mutual_info résultat")
print('-----------------------------------------------------')
print(feature_names_mutual_info)
print('-----------------------------------------------------')

print("shape of the matrix before applying the embedded feature selection:", News.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
model = SelectFromModel(lsvc).fit(News,
                                  y_train)  # you can add threshold=0.18 as another argument to select features that have an importance of more than 0.18
X_new = model.transform(News)
print("shape of the matrix after applying the embedded feature selection:", X_new.shape)
print(model.get_support())
print("Features selected by SelectFromModel: ", feature_names[model.get_support()])

Decision_Tree = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('classification', DecisionTreeClassifier())
])

Decision_Tree.fit(X_train, y_train)
target_category = data['category'].unique()
y_pred1 = Decision_Tree.predict(X_test)
print("le résultat du model avant l'utilisation de Features selection")
print('---------------------------------------------')
print(metrics.classification_report(y_test, y_pred1, target_names=target_category))
print('---------------------------------------------')


DecisionTree_Chi2 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectKBest(chi2, k=20)),
    ('classification', DecisionTreeClassifier())
])
DecisionTree_Chi2.fit(X_train, y_train)

y_pred3 = DecisionTree_Chi2.predict(X_test)
print("le résultat du model après l'utilisation de Features selection chi2")
print('---------------------------------------------')
print(metrics.classification_report(y_test, y_pred3, target_names=target_category))
print('---------------------------------------------')

M_NB = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
    ('classification', MultinomialNB(alpha=0.01))
])
M_NB.fit(X_train, y_train)
y_pred5 = M_NB.predict(X_test)
print("le résultat du model NB avant l'utilisation de Features selection")
print('----------------------------------------------------------')
print(metrics.classification_report(y_test, y_pred5, target_names=target_category))
print('----------------------------------------------------------')

M_NB1 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectKBest(chi2, k=200)),
    ('classification', MultinomialNB())
])
M_NB1.fit(X_train, y_train)
y_pred4 = M_NB1.predict(X_test)
print("le résultat du model NB après l'utilisation de Features selection chi2")
print('----------------------------------------------------------')
print(metrics.classification_report(y_test, y_pred4, target_names=target_category))
print('----------------------------------------------------------')



M_SVM = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', TruncatedSVD(n_components=15, random_state=321)),
    ('classification', svm.SVC())
])
M_SVM.fit(X_train, y_train)
y_pred6 = M_SVM.predict(X_test)
print("le résultat du model SVM avant l'utilisation de Features selection")
print('----------------------------------------------------------')
print(metrics.classification_report(y_test, y_pred6, target_names=target_category))
print('----------------------------------------------------------')

M_SVM1 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectKBest(chi2, k=200)),
    ('classification', svm.SVC())
])
M_SVM1.fit(X_train, y_train)
y_pred7 = M_SVM1.predict(X_test)
print("le résultat du model après l'utilisation de Features selection chi2")
print('----------------------------------------------------------')
print(metrics.classification_report(y_test, y_pred7, target_names=target_category))
print('----------------------------------------------------------')

Decision_Tree = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('classification', DecisionTreeClassifier())
])

Decision_Tree.fit(X_train, y_train)
target_category = data['category'].unique()
y_pred1D = Decision_Tree.predict(X_test)
print("le résultat du model DT après l'utilisation de Features selection f_classif")
print('---------------------------------------------')
print(metrics.classification_report(y_test, y_pred1D, target_names=target_category))
print('---------------------------------------------')

NB = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('classification', svm.SVC())
])

NB.fit(X_train, y_train)
target_category = data['category'].unique()
y_pred1N = NB.predict(X_test)
print("le résultat du model NB après l'utilisation de Features selection f_classif")
print('---------------------------------------------')
print(metrics.classification_report(y_test, y_pred1N, target_names=target_category))
print('---------------------------------------------')

SVM = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('feature_extraction', TfidfTransformer()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('classification', svm.SVC())
])

SVM.fit(X_train, y_train)
target_category = data['category'].unique()
y_pred1S = SVM.predict(X_test)
print("le résultat du model SVM après l'utilisation de Features selection f_classif")
print('---------------------------------------------')
print(metrics.classification_report(y_test, y_pred1S, target_names=target_category))
print('---------------------------------------------')

