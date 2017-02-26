import pandas as pd
import preprocess as pp
import matplotlib.pyplot as plt

from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, train_test_split
from mlxtend.evaluate import confusion_matrix 
from mlxtend.plotting import plot_confusion_matrix

def main():

    model = RandomForestClassifier() 
    cross_val_eval(model)
    confusion_matrix_eval(model)


def cross_val_eval(model):

    df = pp.preprocess_data()

    training_data = df.drop('Survived', axis=1)
    training_labels = df['Survived'] 

    split_type = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

    scores = cross_val_score(model, 
            X = training_data, y = training_labels, 
            cv = split_type)

    print("Simple Eval: ")
    print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix_eval(model):

    df = pp.preprocess_data()

    data = df.drop('Survived', axis=1).as_matrix()
    labels = df['Survived'].as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.4, random_state=0)

    model = model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    cm = confusion_matrix(y_target=y_test, 
                                  y_predicted=predictions)

    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()


if __name__ == "__main__":
    main()
