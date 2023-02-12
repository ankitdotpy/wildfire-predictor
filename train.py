import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import tensorflow as tf

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Train():
	def __init__(self,dataset_path,save_path):
		self.dataset_path = dataset_path
		self.save_path = save_path

	def train(self,model):
		gen = tf.keras.preprocessing.image.ImageDataGenerator()
		# print(dataset_path)
		train_gen = gen.flow_from_directory(
			directory = self.dataset_path+'/train',
			batch_size=32,
			class_mode='categorical')

		val_gen = gen.flow_from_directory(
			directory = self.dataset_path+'/valid',
			batch_size=32,
			class_mode='categorical')

		model.compile(
			'Adam',
			loss = tf.keras.losses.CategoricalCrossentropy(),
			metrics = ['accuracy']
			)

		model.fit(
			x = train_gen,
			validation_data = val_gen,
			batch_size = 32,
			epochs = 5,
			shuffle = True
			)

		model.save(self.save_path+'/final_model.h5')

	def test():
		pass