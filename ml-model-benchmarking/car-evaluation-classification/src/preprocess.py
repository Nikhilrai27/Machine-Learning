import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def split_feature_target(df):
    x=df.drop("class",axis=1)
    y=df['class']
    return x,y

def encode_and_split(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2,stratify=y)
    encoder=OneHotEncoder(handle_unknown="ignore")
        #If a category appears in test but not in train,it won’t crash.

    x_train_encoded=encoder.fit_transform(x_train)
    x_test_encoded=encoder.transform(x_test)

    return x_train_encoded,x_test_encoded,y_train,y_test,encoder