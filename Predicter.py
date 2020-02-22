#学習したモデルを元に実際に未知のニュースタイトルのジャンルを予測する
import Trainer
import PreProcesser
import chainer
import chainer.links as L

model = L.Classifier(Trainer.nn)

chainer.serializers.load_npz("model/model.net",model)

title = input(u"予測したいニュースタイトルを入力してください。")

title_tokenized = PreProcesser.Tokenize_SingleText(title)

title_vectorised = Trainer.model_doc2vec.infer_vector(title_tokenized)

print(model.predictor(title_vectorised))