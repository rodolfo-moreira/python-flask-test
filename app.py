import os
import logging
import socket
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import tensorflow as tl
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import fashion_mnist
#from tensorflow.keras.models import

HOST_NAME = os.environ.get('OPENSHIFT_APP_DNS', 'localhost')
APP_NAME = os.environ.get('OPENSHIFT_APP_NAME', 'flask')
IP = os.environ.get('OPENSHIFT_PYTHON_IP', '127.0.0.1')
PORT = int(os.environ.get('OPENSHIFT_PYTHON_PORT', 8080))
HOME_DIR = os.environ.get('OPENSHIFT_HOMEDIR', os.getcwd())

log = logging.getLogger(__name__)
app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        'host_name': HOST_NAME,
        'app_name': APP_NAME,
        'ip': IP,
        'port': PORT,
        'home_dir': HOME_DIR,
        'host': socket.gethostname()
    })

@app.route('/tensorflow')
def tensorflow():
    df = pd.read_csv("data/titanic.csv")
    df.head(7)
    train_df = df.drop(["PassengerId","Name", "Ticket"], axis=1)

    male_mean_age = train_df[train_df["Sex"]=="male"]["Age"].mean()
    female_mean_age = train_df[train_df["Sex"] == "female"]["Age"].mean()

    print("female mean age: %1.0f" % female_mean_age)
    print("male mean age: %1.0f" % male_mean_age)

    return app.response_class(train_df.to_json(), content_type='application/json')


@app.route('/keras')
def keras():
    dataset = np.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, epochs=150, batch_size=10)

    scores = model.evaluate(X, Y)

    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )

    return "OK"

@app.route('/model')
def model():
    #dataset = keras.datasets.fashion_mnist
    ((imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste)) = fashion_mnist.load_data()
    imagens_treino = imagens_treino / 255.0
    modelo = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(1024, activation=tl.nn.relu),
        Dense(512, activation=tl.nn.relu),
        Dense(256, activation=tl.nn.relu),
        Dense(128, activation=tl.nn.relu),
        Dropout(0.1),
        Dense(10, activation=tl.nn.softmax)
    ])

    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    modelo.fit(imagens_treino, identificacoes_treino, epochs=15, batch_size=128, validation_split=0.2)

    score = modelo.evaluate(imagens_treino, identificacoes_treino, batch_size=128)

    print(score)

    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

#app.run(debug=True)