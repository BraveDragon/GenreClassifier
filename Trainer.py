# coding:"utf_8"
import numpy as np
import pickle
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import training
from chainer.optimizer_hooks.weight_decay import WeightDecay
class NN(chainer.Chain):
    
    def __init__(self, *n_units, n_out = 8):

        self.units = n_units
        super(NN, self).__init__() 
        with self.init_scope():
            self.fc = []
            self.bn = []
            for unit in n_units:
                self.bn.append(L.BatchNormalization(unit)) 
                self.fc.append(L.Linear(unit))

            self.out = L.Linear(n_out)
        

    def forward(self,x):
        n = len(self.units) 
        h = self.bn[0](x)
        h = self.fc[0](h)
        h = F.relu(h)

        for y in range(1,n):
            h = self.bn[y-1](h)
            h = self.fc[y](h)
            h = F.relu(h)
        
        
        
        h = self.out(h)
        h = F.softmax(h)

        return h

#PKLファイルを指定して読み込む
filename = "news.pkl"
titletokens_vectorized = []
Categories = []

with open(filename,mode="rb") as f:
    articles = pickle.load(f)
    titletokens_vectorized = articles.news
    Categories = articles.category

x = np.array(titletokens_vectorized).astype('f')
t = np.array(Categories).astype('i')
InputDim = x.shape[1]

nn = NN(InputDim,200,100,n_out=8)

def main():
    global nn
    global x
    global t
    global InputDim
    
    np.random.seed(0)

    model = L.Classifier(nn)

    dataset = list(zip(x,t))
    n_train = int(len(dataset)*0.7)
    train, test = chainer.datasets.split_dataset_random(dataset, n_train, seed=0)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for param in nn.params():
        if param.name != 'b':
            param.update_rule.add_hook(WeightDecay(0.005))

    batch_size = 100
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)

    epoch = 100

    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result/genre')

    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))

    trainer.extend(extensions.PrintReport(['epoch','main/accuracy','validation/main/accuracy',
                                           'main/loss','validation/main/loss','elapsed_time']),trigger=(1,'epoch'))

    trainer.run()

    #学習したモデルを保存
    chainer.serializers.save_npz("model/model.net", model)

if __name__ == "__main__":
    main()
