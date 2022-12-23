import tensorflow as tf
import os
import argparse
# Convert the keras model.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def main():
    parser = argparse.ArgumentParser(description = 'Parser to evaluate TFLite model')
    parser.add_argument('--model_path', type=str, required = True, help = '')
    parser.add_argument('--save_path', type=str, required = True, help = '')
    parser.add_argument('--model_name', type=str, required = True, help = '')
    
    args = parser.parse_args()

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    model_name = args.model_path
    model = tf.keras.models.load_model(model_name+"_basic.h5", compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    # Save the tflite model.
    with open(args.save_path+"/"+args.model_name +'.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main()