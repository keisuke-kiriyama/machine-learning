import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScalar, StandardScalar
import numpy as np
from sklearn.cross_validation import train_test_split


def create_csv_data_with_nan():
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0, 6.0,,8.0
    10.0,11.0,12.0'''

    df = pd.read_csv(StringIO(csv_data))
    return df


def completion_nan_of_data():
    df = create_csv_data_with_nan()
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    imputed_data = pd.DataFrame(imputed_data)
    return imputed_data


def process_category_data():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
    df.columns = ['color', 'size', 'price', 'classlabel']
    size_mapping = {'XL':3, 'L':2, 'M':1}
    class_mapping = {label:idx for idx, label in
                     enumerate(np.unique(df['classlabel']))}
    #inv_size_mapping = {v:k for k, v in size_mapping.items()}
    #inv_class_mapping = {v:k for k, v in class_mapping.items()}
    df['size'] = df['size'].map(size_mapping)
    df['classlabel'] = df['classlabel'].map(class_mapping)

    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    #class_le.inverse_transform(y)

    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])

    ohe = OneHotEncoder(categorical_features=[0], sparse=False)
    #print(ohe.fit_transform(X))

    print(pd.get_dummies(df[['price', 'color', 'size']]))


def split_test_train_data():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                          header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of Ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    mms = MinMaxScalar()
    stdsc = StandardScalar()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mmx.transform(X_test)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)




split_test_train_data()