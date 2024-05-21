import numpy as np
import os
import tensorflow as tf
import random
from http import HTTPStatus
from PIL import Image
from flask import Flask, jsonify, request
from google.cloud import storage
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
os.environ['TF_CPP_MAIN_LOG_LEVEL'] = '3'

load_dotenv()

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_CLASSIFICATION'] = './models/model.h5'
app.config['GCS_CREDENTIALS'] = './credentials/gcs.json'

model_classification = tf.keras.models.load_model(
    app.config['MODEL_CLASSIFICATION'], compile=False)

bucket_name = os.environ.get('BUCKET_NAME', 'ml-deploy-110803')
client = storage.Client.from_service_account_json(
    json_credentials_path=app.config['GCS_CREDENTIALS'])
bucket = storage.Bucket(client, bucket_name)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


classes = ['Coris julis', 'Trigloporus lastoviza', 'Mugil cephalus']


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': "hello world"
    }), HTTPStatus.OK


@app.route('/predict', methods=['POST'])
def predictions():
    if request.method == 'post':
        req_image = request.files['image']
        if req_image and allowed_file(req_image.filename):
            filename = secure_filename(req_image.filename)
            req_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            img = Image.open(image_path).convert('RGB')
            img = img.resize((160, 160))

            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 225

            classification_result = model_classification.predict(
                x, batch_size=1)
            result = {
                'class': classes[np.argmax(classification_result)],
                'probability': str(np.max(classification_result))
            }
            # array -1 digunakan untuk mengambil nilai array paling terakhir
            image_name = image_path.split('/')[-1]
            blob = bucket.blob(
                'images/' + str(random.randint(10000, 999999))+image_name)
            blob.upload_from_filename(image_path)
            os.remove(image_path)

            return jsonify({
                'status': {
                    'code': HTTPStatus.OK,
                    'message': 'Success predicting',
                    'data': result
                }
            }), HTTPStatus.OK
        else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'File extension not allowed'
                }
            }), HTTPStatus.BAD_REQUEST
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.METHOD_NOT_ALLOWED,
                'message': 'Method not allowed'
            }
        }), HTTPStatus.METHOD_NOT_ALLOWED


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
