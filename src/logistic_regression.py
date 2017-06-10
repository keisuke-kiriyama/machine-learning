from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from perceptron import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def get_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split( \
        X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def get_data_std():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split( \
        X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test


def create_combined_data(X_train_std, X_test_std, y_train, y_test):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    return X_combined_std, y_combined


def skl():
    X_train_std, X_test_stc, y_train, y_test = get_data_std()

    ppn=Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std, y_combined = create_combined_data(X_train_std, X_test_std, y_train, y_test)
    plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                          test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def check_sigmoid():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.yticks([0.0,0.5,1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.show()


def logistic_regression():
    X_train_std, X_test_std, y_train, y_test = get_data_std()
    X_combined_std, y_combined = create_combined_data(X_train_std, X_test_std, y_train, y_test)

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

def check_C():
    X_train_std, X_test_std, y_train, y_test = get_data_std()
    weights, params = [], []
    for c in np.arange(-5, 5):
        lr = LogisticRegression(C=10**c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='patal length')
    plt.plot(params, weights[:, 1], linestypee='--', label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()










