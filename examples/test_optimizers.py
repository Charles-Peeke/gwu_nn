import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense
from gwu_nn.activation_layers import Sigmoid, RELU
from gwu_nn.optimizers import SGD, Adagrad, RMSprop, Adam

y_col = 'Survived'
x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df = pd.read_csv('titanic_data.csv')
y = np.array(df[y_col]).reshape(-1, 1)
orig_X = df[x_cols]

# Lets standardize our features
scaler = preprocessing.StandardScaler()
stand_X = scaler.fit_transform(orig_X)
X = stand_X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

network = GWUNetwork()
network.add(Dense(14, True, input_size=6))
network.add(Dense(28, True))
network.add(Dense(1, True, activation='sigmoid', input_size=14))
network.compile(loss='log_loss', lr=1, optimizer=SGD(lr=1))
network.fit(X_train, y_train, batch_size=10, epochs=50)

