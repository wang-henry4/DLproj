import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pickle

root_path = "/Users/whenry/stuff/DPproject/"
train_csv_path = root_path + "data/train.csv"
img_path = root_path + "train/"
classes = 28

def get_label(line):
    label = tf.strings.split(line, ',')[-1]
    label = tf.strings.split(label, ' ')
    label = tf.strings.to_number(label, out_type=tf.int32)
    label = tf.reduce_sum(tf.one_hot(label, classes), axis=0)
    return label

def get_file(line):
    path = tf.strings.split(line, ',')[0]
    path = tf.strings.join([img_path, path, ".png"])
    return path
    
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
    return tf.image.resize(img, [224, 224])


def process_line(line):
    label = get_label(line)
    img_path = get_file(line)
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    return img, label    

def get_model(transfer_layer):
    model = ResNet50V2(weights="imagenet", include_top = False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
        
    X = model.get_layer(transfer_layer).output
    X = tf.keras.layers.Conv2D(64, 1)(X)
    x = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation("relu")(X)
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    X = tf.keras.layers.Dense(classes, activation='sigmoid')(X)
    new_model = tf.keras.Model(inputs=model.input, outputs=X)
    
    Input_Layer = tf.keras.layers.Input(shape = (224, 224, 3))
    In = tf.keras.layers.BatchNormalization()(Input_Layer)
    out = new_model(In)
    new_model = tf.keras.Model(inputs=Input_Layer, outputs=out)
    return new_model

data = tf.data.TextLineDataset(train_csv_path)
data = data.map(process_line)
data = data.repeat(10).shuffle(128).batch(32)

for t_layer in ["conv2_block3_out", "conv3_block3_out", "conv4_block3_out", "conv5_block3_out"]:
    model = get_model(t_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    history = model.fit(data, epochs=20, steps_per_epoch=826)
    model.save(root_path+"models/{t_layer}.h5")
    with open(root_path+t_layer+"_history", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)