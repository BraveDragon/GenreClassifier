# coding:"utf_8"
import Trainer
import pickle
import chainer
import chainer.links as L
import pprint
import numpy as np

domestic_news = []
world_news = []
business_news = []
entertainment_news = []
sports_news = []
it_news = []
science_news = []
local_news = []

#PKLファイルを指定して読み込む
filename = "news.pkl"
titletokens_vectorized = []
Categories = []

model = L.Classifier(Trainer.nn)
#chainerのモデルを読み込む
chainer.serializers.load_npz("model/model.net",model)
print(" I read the chainer model.")

with open(filename,mode="rb") as f:
    articles = pickle.load(f)
    titletokens_vectorized = articles.news
    Categories = articles.category

x = np.array(titletokens_vectorized).astype('f')
t = np.array(Categories).astype('i')
InputDim = x.shape[1]
dataset = list(zip(x,t))

#ニュースジャンルごとの総数を求める
whole_domestic_news = [len(t[t == 0]),0,0,0,0,0,0,0,0]
whole_world_news = [len(t[t == 1]),0,0,0,0,0,0,0,0]
whole_business_news = [len(t[t == 2]),0,0,0,0,0,0,0,0]
whole_entertainment_news = [len(t[t == 3]),0,0,0,0,0,0,0,0]
whole_sports_news = [len(t[t == 4]),0,0,0,0,0,0,0,0]
whole_it_news = [len(t[t == 5]),0,0,0,0,0,0,0,0]
whole_science_news = [len(t[t == 6]),0,0,0,0,0,0,0,0]
whole_local_news = [len(t[t == 7]),0,0,0,0,0,0,0,0]

whole_domestic_news_vectorized = []
whole_world_news_vectorized = []
whole_business_news_vectorized = []
whole_entertainment_news_vectorized = []
whole_sports_news_vectorized = []
whole_it_news_vectorized = []
whole_science_news_vectorized = []
whole_local_news_vectorized = []

#ジャンルごとの分類
for t2 in range(len(t)):
    if t[t2] == 0:
        whole_domestic_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 1:
        whole_world_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 2:
        whole_business_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 3:
        whole_entertainment_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 4:
        whole_sports_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 5:
        whole_it_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 6:
        whole_science_news_vectorized.append(x[t2])

for t2 in range(len(t)):
    if t[t2] == 7:
        whole_local_news_vectorized.append(x[t2])

whole_domestic_news_vectorized = np.array(whole_domestic_news_vectorized)
whole_world_news_vectorized = np.array(whole_world_news_vectorized)
whole_business_news_vectorized = np.array(whole_business_news_vectorized)
whole_entertainment_news_vectorized = np.array(whole_entertainment_news_vectorized)
whole_sports_news_vectorized = np.array(whole_sports_news_vectorized)
whole_it_news_vectorized = np.array(whole_it_news_vectorized)
whole_science_news_vectorized = np.array(whole_science_news_vectorized)
whole_local_news_vectorized = np.array(whole_local_news_vectorized)

whole_domestic_news_vectorized = [x.reshape(1,InputDim) for x in whole_domestic_news_vectorized]
whole_world_news_vectorized = [x.reshape(1,InputDim) for x in whole_world_news_vectorized]
whole_business_news_vectorized = [x.reshape(1,InputDim) for x in whole_business_news_vectorized]
whole_entertainment_news_vectorized = [x.reshape(1,InputDim) for x in whole_entertainment_news_vectorized]
whole_sports_news_vectorized = [x.reshape(1,InputDim) for x in whole_sports_news_vectorized]
whole_it_news_vectorized = [x.reshape(1,InputDim) for x in whole_it_news_vectorized]
whole_science_news_vectorized = [x.reshape(1,InputDim) for x in whole_science_news_vectorized]
whole_local_news_vectorized = [x.reshape(1,InputDim) for x in whole_local_news_vectorized]

for news in whole_domestic_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_domestic_news[predict] = whole_domestic_news[predict] + 1

for news in whole_world_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_world_news[predict] = whole_world_news[predict] + 1

for news in whole_business_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_business_news[predict] = whole_business_news[predict] + 1

for news in whole_entertainment_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_entertainment_news[predict] = whole_entertainment_news[predict] + 1

for news in whole_sports_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_sports_news[predict] = whole_sports_news[predict] + 1

for news in whole_it_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_it_news[predict] = whole_it_news[predict] + 1

for news in whole_science_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_science_news[predict] = whole_science_news[predict] + 1

for news in whole_local_news_vectorized:
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        predict = model.predictor(news)
    predict = np.argmax(predict[0,:].array) + 1
    whole_local_news[predict] = whole_local_news[predict] + 1

#結果のプリント処理
pprint.pprint(whole_domestic_news)
pprint.pprint(whole_world_news)
pprint.pprint(whole_business_news)
pprint.pprint(whole_entertainment_news)
pprint.pprint(whole_sports_news)
pprint.pprint(whole_it_news)
pprint.pprint(whole_science_news)
pprint.pprint(whole_local_news)


