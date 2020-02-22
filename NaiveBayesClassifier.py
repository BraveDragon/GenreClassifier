# coding: "utf-8"
#ナイーブベイズ法による分類を行う
from sklearn import naive_bayes, model_selection
import joblib
import numpy as np
import pickle
import pandas as pd

#PKLファイルを指定して読み込む
filename = input(u'ファイル名を入力してください。')
titletokens_vectorized = []
Categories = []

with open(filename,mode="rb") as f:
    articles = pickle.load(f)
    titletokens_vectorized = articles.news
    Categories = articles.category


x = np.array(titletokens_vectorized).astype('f')
t = np.array(Categories).astype('i')

Classifier = naive_bayes.MultinomialNB(0.1,True)

np.random.seed(0)
TrainSize = 0.6
x_train, x_test = model_selection.train_test_split(x, train_size=TrainSize)
t_train, t_test = model_selection.train_test_split(t, train_size=TrainSize)

Classifier.fit(x_train,t_train)
print(Classifier.score(x_train,t_train))
print(Classifier.score(x_test,t_test))
joblib.dump(Classifier,"NaiveBayesModel.pkl",compress=True)