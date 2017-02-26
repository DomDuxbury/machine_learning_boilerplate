import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper

def main():
    df = preprocess_data()
    print df

def preprocess_data():
    df = _read_data()
    df = _engineer_features(df)
    df = _vectorise_and_scale(df)
    return df

def _read_data():
    # Read data in and drop incomplete rows
    df = pd.read_csv('./data/titanic.csv')
    df = df.dropna()
    return df

def _engineer_features(df):
    # Example of engineering a feature from existing
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySize'] = df['FamilySize'].astype(float)
    return df

def _vectorise_and_scale(df):
    # Process features 
    # Vectorise labels
    # Scale continuous values 
    mapper = DataFrameMapper([
        ('Sex', LabelBinarizer()),
        ('Pclass', LabelBinarizer()),
        ('Survived', None),
        (['Age'], StandardScaler()),
        (['FamilySize'], StandardScaler()),
        (['Fare'], StandardScaler())
    ], df_out = True)

    # Apply the transform
    return mapper.fit_transform(df)

if __name__ == "__main__":
    main()
