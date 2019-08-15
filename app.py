from flask import Flask
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

@app.route('/')
def index():
    return 1

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()