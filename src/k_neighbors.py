from logistic_regression import get_data_std, create_combined_data
from perceptron import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def k_neighbor_classify():
    X_train_std, X_test_std, y_train, y_test = get_data_std()
    X_combined_std, y_combined = create_combined_data(X_train_std, X_test_std, y_train, y_test)
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

k_neighbor_classify()