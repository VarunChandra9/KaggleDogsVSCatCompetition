from flask import Flask, render_template, request 
from werkzeug.utils import secure_filename

import cv2
import tensorflow as tf


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



app = Flask(__name__)


@app.route('/')
def upload_file1():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        model = tf.keras.models.load_model("Final-3conv-128nodes-0dense")
        prediction = model.predict([prepare(f.filename)])
        if prediction.reshape(1)[0] >= 0.5 :
            return 'Its a dog'
        else :
            return 'Its a cat'

if __name__ == '__main__':
    app.run(debug = True)
