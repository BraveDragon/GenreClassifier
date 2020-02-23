# GenreClassifier
Yahooニュースからスクレイピングしてきたニュース記事のタイトルから、そのニュースがどのジャンルに属するかを判定するプログラムです。
### 使用したPythonのバージョン：3.7.5

## ＜使用したライブラリ＞
 - numpy
 - chainer
 - scikit-learn
 - pandas
 - MeCab
 - gensim
 - joblib
 
 ### MeCabについて
 MeCabを利用するためには、ライブラリ本体に加え、MeCabの導入が必要です。導入していない方は以下のＵＲＬから導入して下さい。
 URL : https://taku910.github.io/mecab/#download
   
   
 
## ＜内容＞
<dl>
  <dt> ConfusionMatrix.py：</dt>
    <dd>「Trainer.py」によって生成されたモデル精度の検証のための混合行列を生成するプログラムです。普通に実行すると結果が一瞬表示されてすぐ消えてしまうので、コマンドプロンプトやIDE等から実行して下さい。</dd>
  <dt> CountVectorizer.py：</dt>
    <dd> scikit-learnのCountVectorizerを利用し、ニュース記事のタイトルをベクトル化するプログラムです。結果はPKLファイルで保存されます。</dd>
  <dt> NaiveBayesClassifier.py：</dt>
    <dd>ナイーブベイズ法を用いて予測モデルを生成するプログラムです。実行後、生成されたモデルがPKLファイルで保存されます。</dd>
  <dt> NaiveBayesConfusionMatrix.py：</dt>
    <dd>「NaiveBayesClassifier.py」によって生成されたモデルの精度検証のための混合行列を生成するプログラムです。普通に実行すると結果が一瞬表示されてすぐ消えてしまうので、コマンドプロンプトやIDE等から実行して下さい。</dd>
  <dt> Predicter.py：</dt>
    <dd>「Trainer.py」によって生成されたモデルを元に実際に未知のニュースタイトルのジャンルを予測するプログラムです。普通に実行すると結果が一瞬表示されてすぐ消えてしまうので、コマンドプロンプトやIDE等から実行して下さい。</dd>
　<dt> PreProcesser.py：</dt>
    <dd>ベクトル化の前に必要な前処理用のメソッドをまとめたファイルです。</dd>
  <dt> TFIDFVectorizer.py：</dt>
    <dd>scikit-learnのTfidfVectorizerを利用し、ニュース記事のタイトルをベクトル化するプログラムです。結果はPKLファイルで保存されます。</dd>
  <dt> Trainer.py：</dt>
    <dd>ニューラルネットワークを用いて予測モデルを生成するプログラムです。実行後、「model」フォルダが生成され、その中にモデルがNETファイルで保存されます。</dd>
  <dt> Vectorizer.py：</dt>
    <dd>Doc2Vecを利用してニュース記事をベクトル化するプログラムです。結果はPKLファイルで保存されます。実行の際は</dd>
</dl>

## ＜このプログラムで使用しているDoc2Vecモデルについて＞
　 このプログラムはニュース記事タイトルをベクトル化するため、奥田 裕樹 様のDoc2Vecモデル「dbow300d」を利用しています。  
　 日本語Wikipediaで学習したdoc2vecモデル - Out-of-the-box  URL：https://yag-ays.github.io/project/pretrained_doc2vec_wikipedia/  
　 Doc2Vecモデルのダウンロードページ：URL：https://www.dropbox.com/s/j75s0eq4eeuyt5n/jawiki.doc2vec.dbow300d.tar.bz2?dl=0  
　 このDoc2Vecモデルは「CC-BY-SA: Creative Commons Attribution-ShareAlike License」でライセンスされています。  
   ライセンスの詳細につきましてはこちらのサイトをご覧下さい。  
　 Creative Commons — Attribution-ShareAlike 3.0 Unported — CC BY-SA 3.0  
   URL：https://creativecommons.org/licenses/by-sa/3.0/
　 Doc2Vecモデル自体の著作権表示は以下になります。  
　 Copyright © 2018 yag_ays

## ＜ライセンス＞
   このプログラムはCC BY-SA 3.0で提供されています。使用の際は以下のサイトをご覧いただき、ライセンスに従ってご利用下さい。
   Creative Commons — Attribution-ShareAlike 3.0 Unported — CC BY-SA 3.0  
   URL：https://creativecommons.org/licenses/by-sa/3.0/

## ＜精度＞
  ### ニューラルネットワークモデル
main/loss: 2.0007363855838776  
  main/accuracy: 0.2768749985843897  
  validation/main/loss: 2.04718941450119  
  validation/main/accuracy: 0.17500000074505806  
  
  ### ナイーブベイズモデル
  (TfidfVectorizerのベクトル使用時)  
  main/accuracy: 0.31642857142857145  
  validation/main/accuracy: 0.17237687366167023  

  (CountVectorizerのベクトル使用時)
  main/accuracy: 0.31357142857142856  
  validation/main/accuracy: 0.17130620985010706

思ったような成果を上げられませんでした。

## ＜考察＞
　今回、私が思うような精度を実現できなかったのは以下のような原因が考えられます。
  1. タイトルだけでは入力の情報量としては足りなかった
  今回、予測モデルの入力としてニュースのタイトルを利用していましたが、それでは情報不足だったのかもしれないと思いました。よく似たタイトル名の記事が別のジャンルに分類されていたりしたので、それが原因でジャンルの分類が上手くいかなかった可能性があります。
       
  2. ジャンルごとのニュース記事の量が偏っていた
  データセットがYahooニュースの記事からタイトルをスクレイピングしたものであるため、元のニュース記事のジャンルごとの偏りが過学習の原因の一つになってしまったのではないかと思います。事実、カテゴリ「domestic」のニュースは443件あるのに対し、「it」は120件、「science」は163件というようにジャンルごとのニュースの件数に偏りが生じていました。 また、学習済モデルで混合行列を作成したところ、ニュースの件数が多いジャンルに出力結果が偏るという傾向が見られました。
  
  ### 改善策
  現時点で思いつく改善策をいくつか挙げておきます。

  1. ニュース記事本文も合わせて入力とする
  これは __＜考察＞__ の1.に対応したもので、発想としては「タイトルだけでは足りないのだから、本文も合わせて学習させれば上手くいくのではないか」ということです。しかし、Yahooニュース記事本文のスクレイピングは難易度が高く(本文をスクレイピングするには、タイトルが並んでいるページから、更に2回リンクをたどらなければならない)、現時点では無理かなと思っています。

  2. ニュースの件数が少ないジャンルに関してスクレイピングを行い、ジャンルごとのニュース件数のバラツキを減らす
  これは __＜考察＞__ の2.に対応したもので、ジャンルごとの偏りをなくすことでニュースの件数が多いジャンルに出力結果が偏るのをなくそうということです。しかし、Yahooニュース側の更新を待たなければならないので、時間がかかってしまいます。
