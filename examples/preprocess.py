import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper

def main():
    # Read data in and drop incomplete rows
    df = pd.read_csv('./data/titanic.csv')
    df = df.dropna()

    # Example of engineering a feature from existing
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

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
    df = mapper.fit_transform(df) 
    print df

if __name__ == "__main__":
    main()
