import flask
from flask import request, jsonify
import cv2
import numpy as np
from PIL import Image
from keras import models
import time
import imutils
import base64
import datetime
from mtcnn import MTCNN
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# global graph 
# graph = tf.compat.v1.get_default_graph()

date_right_now = datetime.datetime.now()

app = flask.Flask(__name__)

detector_net = cv2.dnn.readNetFromCaffe('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
detector_haar = cv2.CascadeClassifier('face_detector/haarcascade_frontalface_alt.xml')

tflite_model_spoof = models.load_model('checkpoint/dataset20122022_basic.h5')

@app.route('/absence', methods = ['POST','GET'])
def verify():
	if request.method == "GET":
		return jsonify({"response" : "Get Response Success"})
	elif request.method == "POST":
		img = flask.request.form['image']
		idkaryawan = flask.request.form['idkaryawan']
		nik = flask.request.form['nik']
		secret = flask.request.headers.get('secret')

		result = hashlib.sha256(secret.encode())
		hex = result.hexdigest()
		
		if hex != "147e3c03d32d8fd51d90860733df3b6d1ba692614de4d6478451900ac783bf21":
			return 'your secret is invalid'

		imb64 = base64.b64decode(img)
		
		with open("decodeb64/{}_{}.jpg".format(idkaryawan,nik), "wb") as fh:
			fh.write(imb64)

		start = time.time()

		img = cv2.imread("decodeb64/{}_{}.jpg".format(idkaryawan,nik))

		faceCrop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		detector = MTCNN()

		roi = detector.detect_faces(faceCrop)
		x1, y1, width, height = roi[0]['box']

		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		faceCropFinal = img[y1:y2, x1:x2]

		cv2.imwrite("face/face_{}_{}.jpg".format(idkaryawan,nik),faceCropFinal)

		(h, w) = img.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(img, (224,224)), 1.0, (224, 224), (104.0, 177.0, 123.0))

		detector_net.setInput(blob)
		detections = detector_net.forward()
		
		for i in range(0, detections.shape[1]):
			try:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype('int')

				# print(box)

				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)

				face = img[startY:endY, startX:endX]

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
				prediction = tflite_model_spoof.predict(img_array)[0][0]#[0][0]

				print(prediction)

				#Customize this part to your liking...
				if(prediction < 0.5):
					print("Spoof")

					prediction_spoof = "Spoof"

				else :
					print("Real")

					prediction_spoof = "Real"

				end = time.time()
				elapsed_time = (end - start)

				return jsonify(
					{
						"idkaryawan" : idkaryawan,
						"prediction" : prediction_spoof,
						"threshold": str(prediction),
						"elapsed_time" : elapsed_time
					}
				)
			except Exception as e:
				text = '{}\n\n{}'.format(e,date_right_now)
				sender_address = 'spilwanotif@gmail.com'
				sender_pass = 'akcypppwvkgwuxli'
				receiver_address = 'vardyansyahcahya@gmail.com'

				message = MIMEMultipart()
				message['From'] = sender_address
				message['To'] = receiver_address
				message['Subject'] = 'FLASK_FACE_ERROR'

				message.attach(MIMEText(text, 'plain'))

				session = smtplib.SMTP('smtp.gmail.com', 587)
				session.starttls()
				session.login(sender_address, sender_pass)
				text1 = message.as_string()
				session.sendmail(sender_address, receiver_address, text1)
				session.quit()

		

		
		
if __name__ == '__main__':
	app.run(host="0.0.0.0", port=3041)