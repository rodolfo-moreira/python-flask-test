import os
import logging
import socket
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import tensorflow as tl

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

@app.route('/iris')
def teste():
    iris = pd.read_csv("data/Iris.csv")
    iris.head(n=7)
    return app.response_class(iris, content_type='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

#app.run(debug=True)