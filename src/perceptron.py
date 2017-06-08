# coding=utf-8
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd


class Perceptron(object):
    """パーセプトロンの分類器
    パラメータ
    -----------
    eta : float
        学習率(0.0より大きく1.0以下)
    n_iter : int
        トレーニングデータのトレーニング回数
    
    属性
    -----------
    w_ : １次元配列
        適合後の重み
    errors_ : リスト
        各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """トレーニングデータに適合させる
        パラメータ
        ----------
        X : （配列のようなデータ構造), shape = [n_samples, n_features]
            トレーニングデータ
            n_sampleはサンプルの個数、n_featureは特徴量の個数
            
        y : 配列のようなデータ構造, shape = [n_samples]
            目的変数
        
        戻り値
        ----------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter): #トレーニング回数文トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y): #各サンプルで重みを更新
                #重みw1, ... , wmの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                #重みw0の更新
                self.w_[0] += update
                #重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            #反復回数ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    camp = ListedColormap(colors[:len(np.unique(y))])
    #決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #各特徴量を１次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    #グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, camp=camp)
    #値の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl,1], alpha=0.8, c=camp(idx),
                    marker=markers[idx], label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidths=1, marker='',
                    s=55, label='test set')


# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = df.iloc[0:100, [0, 2]].values
# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)
# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel('sepal length [cm]')
# plt.ylabel('patal length [cm]')
# plt.legend(loc='upper left')
# plt.show()
