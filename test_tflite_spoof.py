from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import argparse
import pickle
import cv2
import os
import psycopg2
from datetime import datetime
import requests
import json
# import pysftp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#==================================================================================================================

def load_tflite_model(file):
    interpreter = tf.lite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    return interpreter

def predict(face_model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    sample = np.expand_dims(face_pixels, axis=0)
    input_details = face_model.get_input_details()
    output_details = face_model.get_output_details()
    input_shape = input_details[0]['shape']
    input_data = sample.reshape(input_shape)
    face_model.set_tensor(input_details[0]['index'], input_data)
    face_model.invoke()
    output_data = face_model.get_tensor(output_details[0]['index'])	
    return output_data[0]

def load_image(data_dir, size):
    imgs = []
    print("Load image from ", data_dir, "...")
    for filename in tqdm(glob(data_dir + '/*')):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = np.asarray(img)
        img = img / 255
        imgs.append(img)
    return np.asarray(imgs)

def load_data(data_dir, img_size):
    fake_data_dir = os.path.join(data_dir, 'spoof')
    fake_filenames = glob(fake_data_dir + "/*")
    if len(fake_filenames) == 0:
        print(fake_data_dir, " is empty")
        fake_data = []
        fake_label = []
    else:
        fake_data = load_image(fake_data_dir, img_size)
        fake_label = np.zeros(len(fake_data))
    
    real_data_dir = os.path.join(data_dir, 'real')
    real_filenames = glob(real_data_dir + "/*")
    if len(real_filenames) == 0:
        print(real_data_dir, " is empty")
        real_data = []
        real_label = []
    else:
        real_data = load_image(real_data_dir, img_size)
        real_label = np.ones(len(real_data))
    
    if len(fake_data) == 0:
        return real_data, real_label
    elif len(real_data) == 0:
        return fake_data, fake_label
    else:
        data_test = np.concatenate((real_data, fake_data))
        label_test = np.concatenate((real_label, fake_label))
        
        return data_test, label_test, real_filenames+fake_filenames

def save_pickle(filename, data):
    with open(filename,'wb') as f:
        pickle.dump(data, f)
    print("Data saved in ", filename)

def calculate_accuracy(model_path,img_size,test_dir,threshold):
    model = load_tflite_model(model_path)

    img_size = (img_size)
    data_test, label_test, filenames = load_data(test_dir, img_size)
    
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    probabilities = []
    print("Forward pass using Test Data...")
    for data, label, filename in tqdm(zip(data_test, label_test, filenames)):
        pred = predict(model, data)
        probabilities.append(pred)
        if label == 0:
            if pred < threshold:
                tp += 1
            else:
                print(filename, pred)
                fn += 1
        elif label == 1:
            if pred >= threshold:
                tn += 1
            else:
                print(filename, pred)
                fp += 1


    save_pickle(os.path.basename(model_path.split('.')[0]) + '_probability.pickle', probabilities)

    print("====================================")
    print("Model : ", model_path)
    print("Threshold :", threshold)
    print("True Positive "+ str(tp))
    print("False Negative "+ str(fn))
    print("False Positive "+ str(fp))
    print("True Negative "+str(tn))

    if (tp+fn > 0):
        tpr = tp/(tp+fn)
        print("True Positive Rate :", tpr)
    if (fp+tn > 0):
        fpr = fp/(fp+tn)
        print("False Positive Rate :", fpr)
    
    accuracy = (tpr + (1 - fpr))/2
    f1 = tp/(tp +(fp+fn)/2)
    recall = tp/(tp+fn)
    print("Accuracy :", accuracy, "%")
    print("====================================")
    return accuracy,f1,recall,tp,fp,tn,fn

def main():
    parser = argparse.ArgumentParser(description = 'Parser to evaluate TFLite model')
    parser.add_argument('--model', type=str, required = True, help = 'TFLite model path')
    parser.add_argument('--existing_model', type=str, required = True, help = 'TFLite model path existing')
    parser.add_argument('--test_dir', type=str, required = True, help = 'Data test path consisting of real and fake directory', default=None)
    parser.add_argument('--img_size', type=int, help='Image size', default=224)
    parser.add_argument('--threshold', type=float, help = 'Test batch size', default = 0.5)
    parser.add_argument('--cuda', type = str, help = 'CUDA Device ID', default='1')
    # parser.add_argument('--upload_path', type = str,required = True, help = 'upload_path')

    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    accuracy,f1,recall,tp,fp,tn,fn = calculate_accuracy(args.model,(args.img_size,args.img_size),args.test_dir,args.threshold)
    accuracy_existing_model,f1_existing,recall_existing,tp_existing,fp_existing,tn_existing,fn_existing = calculate_accuracy(args.existing_model,(args.img_size,args.img_size),args.test_dir,args.threshold)
    
    comparasion_result = "SAME"
    

    if accuracy_existing_model<accuracy:
        comparasion_result = "BETTER"
        # with pysftp.Connection('172.20.3.125', username='covid', password='!n24H!nwF0 0k3') as sftp:
        #     with sftp.cd('model'):
        #         sftp.put(args.model,args.upload_path)
        
        #production
        
    elif accuracy_existing_model>accuracy:
        comparasion_result = "WORST"

    # text = "model: {}\n dataset: {}\n accuracy: {}\n f1: {}\n recall: {}\n threshold: {}\n tp: {}\n fp: {}\n tn : {}\n fn : {}\n tipe : {}\ncomparison_result : {}\n".format(model,dataset,accuracy,f1,recall, args.threshold,tp,fp,tn,fn,"spoof_no_mask",comparasion_result)
    # print(text)

    # conn = psycopg2.connect(database="hr_dev", user = "hr", password = "h43r", host = "192.168.1.50", port = "1521")
    # print("Opened database successfully")
    # now = datetime.now()
    # cur = conn.cursor()
    # sql = """INSERT INTO t_auto_train(model_filename, test_set, accuracy,f1_score,recall, threshold,true_positive,false_positive,true_negative,false_negative,type,trained_on,compare_result)
    #         VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    model = args.model.split("/")[-1]
    dataset = args.test_dir.split("/")[-1]
    # cur.execute(sql, (model,dataset,accuracy,f1,recall, args.threshold,tp,fp,tn,fn,"spoof",now,comparasion_result))
    # # get the generated id backs
    # ids = cur.fetchone()[0]
    # print("saved as id "+str(ids))
    # conn.commit()

    # cur.close()

    # conn.close()
	
    # URL_Hangout = "https://chat.googleapis.com/v1/spaces/AAAAfFh4XxY/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=Z00URZevPK7pCkWIzZr9yyQDT8vOEjBSjvqYhND_VBU%3D&thread_key=T8880"
    text = "model: {}\n dataset: {}\n accuracy: {}\n f1: {}\n recall: {}\n threshold: {}\n tp: {}\n fp: {}\n tn : {}\n fn : {}\n tipe : {}\ncomparison_result : {}\n".format(model,dataset,accuracy,f1,recall, args.threshold,tp,fp,tn,fn,"spoof_no_mask",comparasion_result)
    print(text)
    sender_address = 'spilwanotif@gmail.com'
    sender_pass = 'akcypppwvkgwuxli'
    receiver_address = 'vardyansyahcahya@gmail.com'

    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Training Result of Spoof 21122022 vs 20122022'

    message.attach(MIMEText(text, 'plain'))

    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text1 = message.as_string()
    session.sendmail(sender_address, receiver_address, text1)
    session.quit()
    
    print('Mail Sent')

    # message = {'text': text}
    # requests.post(URL_Hangout, data = json.dumps(message))
    
if __name__ == '__main__':
    main()