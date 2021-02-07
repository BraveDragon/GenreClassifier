# coding:"utf_8"
import PreProcesser
import pickle
import gensim

#CSVファイルを指定して読み込む
filename = "news.csv"
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
model_doc2vec = gensim.models.Doc2Vec.load('jawiki.doc2vec.dbow300d.model')
print(" I finished loading the Doc2Vec model.")

ReturnNews = []

for x in titletokens:
    ReturnNews.append(model_doc2vec.infer_vector(x))


ReturnObject = PreProcesser.News(ReturnNews,Categories)
with open(filename+".pkl",mode="wb") as f:
     pickle.dump(ReturnObject,f)
