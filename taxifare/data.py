
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd


def get_data(line_count):
    
    df = pd.read_csv("gs://lewagon_batch_869_thierry/data/train_1k.csv", nrows=line_count)
    
    return df


def clean_df(df):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 1]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def holdout(df):

    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    
    #Testing model
    model = joblib.load('../random_forest.joblib')
    
    import ipdb; ipdb.set_trace()
