import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
print(data.columns)

feature1 = tf.feature_column.numeric_column('2.6487')
feature2 = tf.feature_column.numeric_column('4.5192')

feature_columns = [feature1 , feature2]

xdata = data.drop('1.0000' , axis=1)
labels = data['1.0000']

xtrain , xtest , ytrain , ytest = train_test_split(xdata , labels , test_size=0.3)

input_function = tf.estimator.inputs.pandas_input_fn(x = xtrain , y=ytrain ,
                                                     num_epochs=1000 , batch_size= 10 , shuffle=True)

input_function_test = tf.estimator.inputs.pandas_input_fn(x = xtest , y=ytest ,
                                                     num_epochs=1 , batch_size= 10 , shuffle=False)


model = tf.estimator.DNNClassifier(hidden_units=[10,10,10] , feature_columns=feature_columns , n_classes= 2)


def train_model(input_Fn , step_nums):
    model.train(input_fn=input_Fn , steps=step_nums)

def evalModel(input_fn):
    result = model.evaluate(input_fn)
    print(result)

def predict():
    prediction_input_function = tf.estimator.inputs.pandas_input_fn(x=xtest, batch_size=10, num_epochs=1, shuffle=False)
    prediction = list(model.predict(prediction_input_function))
    print(prediction)
