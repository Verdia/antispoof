import cv2
import numpy as np
from PIL import Image
from keras import models
import imutils
import os
import tensorflow as tf
import time

# checkpoint_path = "training_gender/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
detector_net = cv2.dnn.readNetFromCaffe('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
detector_haar = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')


model = models.load_model('checkpoint/dataset_basic.h5')
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
video = cv2.VideoCapture(0)

time.sleep(2.0)

while True:
        _, frame = video.read()
        frame = imutils.resize(frame, width=600)

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224,224)), 1.0, (224, 224), (104.0, 177.0, 123.0))

        detector_net.setInput(blob)
        detections = detector_net.forward()

        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # print(box)

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]

            try:
                face = cv2.resize(face, (224,224))
            except:
                break
            face = face.astype('float') / 255.0 
            
            # face = tf.keras.preprocessing.image.img_to_array(face)
            img_array = np.array(face)

            #Expand dimensions to match the 4D Tensor shape.
            img_array = np.expand_dims(img_array, axis=0)

            # print(img_array)


            #Calling the predict function using keras
            prediction = model.predict(img_array)[0][0]#[0][0]

            print(prediction)

            #Customize this part to your liking...
            if(prediction < 0.055):
                print("Spoof")

                cv2.putText(frame, "Spoof", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            else :
                print("Real")

                cv2.putText(frame, "Real", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()