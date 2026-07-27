from flask import Flask, request
import os
from Predict import Main
import glob
from Predict_color import MainFunction

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    try:
        for f in glob.glob('images'):
            os.remove(f)
        image_data = request.files['Image']
        image_data.save(f'images/{image_data.filename}')
        img = glob.glob('images/*')[0]
        Type = Main(img)
        Color = MainFunction(img)
        return {'Type': Type, 'Color':Color}
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3377, debug=True)