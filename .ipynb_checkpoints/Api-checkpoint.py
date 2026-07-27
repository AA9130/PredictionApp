from flask import Flask


app = Flask(__name__)


@app.route('/')
def Welcome():
    return 'Welcome'





app.run(host="0.0.0.0", port=3300, debug=True)