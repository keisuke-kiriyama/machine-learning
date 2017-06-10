from logistic_regression import get_data_std, create_combined_data
from sklearn.svm import SVC
from perceptron import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np


def support_vector_machine():
    X_train_std, X_test_std, y_train, y_test = get_data_std()
    X_combined_std, y_combined = create_combined_data(X_train_std, X_test_std, y_train, y_test)

    svm = SVC(kernel='linear', C=10.0, random_state=0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


def create_random_data():
    np.random.seed(0)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    return X_xor, y_xor


def support_vector_machine_rbf():
    X_xor, y_xor = create_random_data()
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.show()


def check_svm_gamma(gamma):
    X_train_std, X_test_std, y_train, y_test = get_data_std()
    X_combined_std, y_combined = create_combined_data(X_train_std, X_test_std, y_train, y_test)
    svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()



check_svm_gamma(10.0)