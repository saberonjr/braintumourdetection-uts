from flask import Flask
from markupsafe import escape

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/<name>')
def hello_name(name):
    return f"<p>Hello, {escape(name)}!</p>"

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    return f"<p>Subpath {escape(subpath)}</p>"