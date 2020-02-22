# coding:"utf_8"
import sys
import PreProcesser
import pickle
from sklearn.feature_extraction.text import CountVectorizer

#CSVファイルを指定して読み込む
filename = input(u'ファイル名を入力してください。')
titletokens = PreProcesser.Tokenize_SplitBySpace(filename,'title')

Vectorizer = CountVectorizer()
X = Vectorizer.fit_transform(titletokens)
X = X.toarray()

Categories = PreProcesser.CategoryToNumber(filename,'genre',\
                                            'domestic',\
                                            'world',\
                                            'business',\
                                            'entertainment',\
                                            'sports',\
                                            'it',\
                                            'science',\
                                            'local'\
                                            )


ReturnNews = []
for x in X:
     ReturnNews.append(x)

ReturnObject = PreProcesser.News(ReturnNews,Categories)
with open(filename+"_Count.pkl",mode="wb") as f:
     pickle.dump(ReturnObject,f)
