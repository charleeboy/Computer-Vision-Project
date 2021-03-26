import base64
from flask import Flask, request, render_template
from PIL import Image, ImageOps
import fastai
import re
import io

app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        image_b64 = request.values['pic']
        image_data = re.sub('^data:image/.+;base64,', '', image_b64)
        image_data2 = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_data2)).convert('RGB')

    return "ok"


if __name__ == '__main__':
    app.run(debug=True)
