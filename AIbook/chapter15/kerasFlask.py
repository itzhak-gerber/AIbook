import flask
import cv2
import numpy as np
import jsonpickle
from flask import Flask, request, Response
from keras.models import model_from_json

classes={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

def Load_model():
    global loaded_model
    # load json and create model
    json_file = open('modelflaskcifer10.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelflaskcifer10.h5")
    
    print("Loaded model from disk")
    orgImg = cv2.imread('dog.jpg')
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGB)
    img=cv2.resize(orgImg,(32,32))
    img=img/255
    sample=img.reshape(1,32,32,3)
    result=loaded_model.predict(sample)
    l=np.argmax(result[0])
    s=classes[l]
    print("loaded model class=",s)

app = flask.Flask(__name__)
print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
Load_model()

def FlaskPredict(orgImg):
    global loaded_model
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGB)
    img=cv2.resize(orgImg,(32,32))
    img=img/255
    sample=img.reshape(1,32,32,3)
    result=loaded_model.predict(sample)
    l=np.argmax(result[0])
    s=classes[l]
    return s

@app.route('/api/predict', methods=['POST'])
def predict():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # do some fancy processing here....
    sClass=FlaskPredict(img)
     # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]),
                'label':sClass
                 }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

app.run(host='0.0.0.0')
