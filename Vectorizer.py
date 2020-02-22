# coding:"utf_8"
import sys
import PreProcesser
import pickle
import gensim

#CSVファイルを指定して読み込む
filename = input(u'ファイル名を入力してください。')
#titletokens = PreProcesser.Tokenize_SplitBySpace(filename,'title')
titletokens = PreProcesser.Tokenize(filename,'title')

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

#doc2vecの学習済みモデルを読み込む
print(" I'm loading the Doc2Vec model.")
model_doc2vec = gensim.models.Doc2Vec.load('ここにDoc2Vecのモデルファイル名を入力(拡張子は.model)')
print(" I finished loading the Doc2Vec model.")

ReturnNews = []

for x in titletokens:
    ReturnNews.append(model_doc2vec.infer_vector(x))


ReturnObject = PreProcesser.News(ReturnNews,Categories)
with open(filename+".pkl",mode="wb") as f:
     pickle.dump(ReturnObject,f)
