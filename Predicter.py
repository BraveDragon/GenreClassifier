#学習したモデルを元に実際に未知のニュースタイトルのジャンルを予測する
import Vectorizer
import Trainer
import PreProcesser
import chainer
import chainer.links as L
import numpy as np

model = L.Classifier(Trainer.nn)

Ansers = ('domestic','world','business','entertainment','sports','it','science','local')

chainer.serializers.load_npz("model/model.net",model)

title = input(u"予測したいニュースタイトルを入力してください。")

title_tokenized = PreProcesser.Tokenize_SingleText(title)[0]

title_vectorised = Vectorizer.model_doc2vec.infer_vector(title_tokenized)
#TODO:警告を直す
#TODO:GPU対応
answerNumber = model.predictor(np.array([title_vectorised]))
answerNumber = np.array(answerNumber.array)
answerNumber = np.argmax(answerNumber)
print(Ansers[answerNumber])