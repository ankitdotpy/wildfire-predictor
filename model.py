import numpy as np
import cv2
import tensorflow as tf

class TopLayer(tf.keras.layers.Layer):
    def __init__(self, classes):
        super(TopLayer, self).__init__()
        self.classes = classes

    def build(self, input_shape):
        self.avgPool = tf.keras.layers.AveragePooling2D(7,7)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1000, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.75)
        self.classify = tf.keras.layers.Dense(self.classes, activation='softmax')

    def call(self, inputs):
        top = self.avgPool(inputs)
        top = self.flatten(top)
        top = self.dense(top)
        top = self.dropout(top)
        top = self.classify(top)
        return top

class Model():
    def __init__(self,num_classes):
        self.num_classes = num_classes

    def model(self):
        base = tf.keras.applications.resnet50.ResNet50(weights='imagenet',include_top=False,input_shape=(350,350,3))
        out = TopLayer(self.num_classes)(base.output)
        mod = tf.keras.models.Model(inputs=base.inputs,outputs=out)

        return mod