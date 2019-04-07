# app.py
from io import BytesIO
import base64
import cv2
from flask import Flask, request, jsonify, make_response
from flask_restplus import Resource, Api, fields
from werkzeug.contrib.fixers import ProxyFix
# from imageio import imread
from PIL import Image
from random import randint
from Oculai import *

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app,
        version='0.1',
        title='Retina',
        description='Oculai Prediction API'
)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

@api.route('/predict')
class Predict(Resource):
    print('==> Prepping')

    model = api.model('Prediction', {'image': fields.String})
    # @api.marshal_with(model)
    @api.expect(model)
    def post(self, **kwargs):
        data =  request.get_json()
        #get data
        img64 = data['image']

        starter = img64.find(',')
        image_data = img64[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        img = BytesIO(base64.b64decode(image_data))
        #.decode('base64')
        # img = imread(io.BytesIO(base64.b64decode(image64)))




        #build model
        model = build_model('C:/Users/Adrian/Desktop/Hackathons/occulai/DiaCNN.h5')
        data['score'] = get_prediction(img, model)
        # data['score'] = get_prediction(input_scaled, model, get_scaler())

        return make_response(jsonify(data))

@app.teardown_appcontext
def shutdown_session(exception=None):
    print('==> Tearing down function');

if __name__ == '__main__':
    app.run(debug=True)
