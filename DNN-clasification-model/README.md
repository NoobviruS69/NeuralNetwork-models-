# Tensorflow DNN-classification
**********
this is a  simple tensorflow dense neural network model to show the tensorflow basics using the tensorflow estimator API.
In order to keep it as simple as possible the dataset used is a simple binary classification csv file with datapoints and 
corresponding (0/1) labels.I wil look into each part of the code for more comprehensive understading.

## Library list:
```
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
```
## Pandas and Feature columns:
First we read the csv file and create feature colums for it
```
data = pd.read_csv('data.csv')
print(data.columns)

feature1 = tf.feature_column.numeric_column('2.6487')
feature2 = tf.feature_column.numeric_column('4.5192')

feature_columns = [feature1 , feature2]u
```
## Feature , Labels and Spliting:
```
xdata = data.drop('1.0000' , axis=1)
labels = data['1.0000']

xtrain , xtest , ytrain , ytest = train_test_split(xdata , labels , test_size=0.3)
```
## Train , Test Input Functions
Next we define the pandas input functions with batchsize = 10(which is unnecessary for such a small dataset)
```
input_function = tf.estimator.inputs.pandas_input_fn(x = xtrain , y=ytrain ,
                                                     num_epochs=1000 , batch_size= 10 , shuffle=True)

input_function_test = tf.estimator.inputs.pandas_input_fn(x = xtest , y=ytest ,
                                                     num_epochs=1 , batch_size= 10 , shuffle=False)
```
## DNN-Model:
Then the actual model comes to place.The fact that we are using the estimator API is laughable being so basic.But for the kind of dataset we have corresponds to that.The purpose of this model is to showcase the aspects that come into picture when dealing with dense neural networks.Thus , here we create the model var and use the DNNClassifier and give **the hidden units(10,10,10) , the feature columns , classes(default=2) , optamizer(default = adagrad) , dropout(default=none)** 
```
model = tf.estimator.DNNClassifier(hidden_units=[10,10,10] , feature_columns=feature_columns , n_classes= 2)
```
## Final output methods:
```
def train_model(input_Fn , step_nums):
    model.train(input_fn=input_Fn , steps=step_nums)

def evalModel(input_fn):
    result = model.evaluate(input_fn)
    print(result)

def predict():
    prediction_input_function = tf.estimator.inputs.pandas_input_fn(x=xtest, batch_size=10, num_epochs=1, shuffle=False)
    prediction = list(model.predict(prediction_input_function))
    print(prediction)

```
## License

**copyright © N00bVrus69**






