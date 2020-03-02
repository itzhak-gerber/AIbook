import requests
import json
import cv2
import sys

def SendImagToEndPoint(fileName):
    addr = 'http://localhost:5000'
    predict_url = addr + '/api/predict'

    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = cv2.imread(fileName)
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(predict_url, data=img_encoded.tostring(), headers=headers)
    # decode response
    print(json.loads(response.text))

def main(argv):
    if len(argv)!=1:
        print("sample")
        print("python client.py dog.jpg")
    else:
        SendImagToEndPoint(argv[0])

if __name__=="__main__":
    main(sys.argv[1:])
