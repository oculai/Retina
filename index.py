# app.py
import io
import base64
from flask import Flask, request, jsonify, make_response
from flask_restplus import Resource, Api, fields
from werkzeug.contrib.fixers import ProxyFix
from imageio import imread
from database import db_session
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
    print('Prepping ->')

    model = api.model('Prediction', {'image': fields.String})
    # @api.marshal_with(model)
    @api.expect(model)
    def post(self, **kwargs):
        data =  request.get_json()
        #get data
        image64 = data['image']
        img = imread(io.BytesIO(base64.b64decode(image64)))

    
        # finally convert RGB image to BGR for opencv
        # and save result
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # input_scaled = scale_data(sensors)
        # input_scaled = input_scaled.reshape(1, 32, 5)
        #input_win = make_windows(input_scaled, 16)
        #input_ready = prep_for_model(input_win, 16, 5)

        #build model
        # model = build_model('C:/Users/Adrian/Desktop/Hackathons/HackABull2019/Models/deep_hive_weights.h5')
        data['score'] = randint(0, 5)
        # data['score'] = get_prediction(input_scaled, model, get_scaler())
        
        return make_response(jsonify(data))

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == '__main__':
    app.run(debug=True)