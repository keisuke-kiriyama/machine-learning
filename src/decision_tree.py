#coding=utf-8
from logistic_regression import get_data, create_combined_data
from perceptron import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

#p(i|t)は特定のノードtに置いてクラスiに属するサンプルの割合

def gini(p):
    return (p)*(1-(p)) + (1-p)*(1-(1-p))

def entropy(p):
    return - p*np.log2(p) - (1-p)*np.log2((1-p))

def error(p):
    return 1 - np.max([p, 1-p])


def check_disorder_method():
    x=np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(i) for i in x]

    fig=plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                              ['Entropy', 'Entropy(scaled)',
                               'Gini Impurity', 'Misclassification Error'],
                              ['-','-','--','-.'],
                              ['black','lightgray','red','green','cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15),
              ncol=3, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    plt.show()


def create_tree_structure(output_file):
    X_train, X_test, y_train, y_test = get_data()
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train,y_train)
    X_combined, y_combined = create_combined_data(X_train, X_test, y_train, y_test)
    plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
    export_graphviz(tree, out_file=output_file, feature_names=['petal length', 'petal width'])
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    #graphviz -> dot -Tpng tree.dot -o tree.png


def create_random_forest_structure():
    X_train, X_test, y_train, y_test = get_data()
    X_combined, y_combined = create_combined_data(X_train, X_test, y_train, y_test)
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


#create_tree_structure('tree.dot')
create_random_forest_structure()
