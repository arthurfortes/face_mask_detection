import silence_tensorflow.auto
from tensorflow.python.keras.models import load_model
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2


__author__ = "Arthur Fortes da Costas"
__copyright__ = "Copyright 2021"
__credits__ = ["Arthur Fortes"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Arthur Fortes"
__email__ = "fortes.arthur@gmail.com@gmail.com"
__status__ = "Dev"


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


keras_model = load_model('model.h5')
cam = cv2.VideoCapture(0) 
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv2.CascadeClassifier(f"{cv2.haarcascades}/{file_name}")


label = {
    0: {"name": "Mask only in the chin", "color": (51, 153, 255), "id": 0},
    1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1},
    2: {"name": "Without mask", "color": (0,0,255), "id": 2},
    3: {"name": "With mask ok", "color": (0, 102, 51), "id": 3},
}


while True:
    status, frame = cam.read()

    if not status:
        break

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray)

    for x,y,w,h in faces:
        color = (0,0,0)
        gray_face = gray[y:y+h+50, x:x+w+50]

        if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:

            gray_face = cv2.resize(gray_face, (300, 300))
            gray_face = gray_face / 255
            gray_face = np.expand_dims(gray_face, axis=0)
            gray_face = gray_face.reshape((1, 300, 300, 1))
            pred = np.argmax(keras_model.predict(gray_face))
            classification = label[pred]["name"]
            color = label[pred]["color"]

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, label[pred]["id"])

            cv2.putText(frame, classification, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{len(faces)} detected face",(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)

    # Mostrando o frame
    cv2.imshow("Cam", frame)
