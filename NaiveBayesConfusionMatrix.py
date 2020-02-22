# coding: "utf-8"
from sklearn import naive_bayes, model_selection, metrics
import joblib

filename_NBmodel = input(u'ナイーブベイズモデルのファイル名を入力してください。')
model = joblib.load(filename_NBmodel)

