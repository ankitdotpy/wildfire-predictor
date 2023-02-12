import cv2
import numpy as np
import tensorflow as tf

class train():
	def __init__(self,dataset_path,save_path):
		self.dataset_path = dataset_path
		self.save_path = save_path

	def train(self,model):
		gen = tf.keras.preprocessing.image.ImageDataGeneretor()

		train_gen = gen.flow_from_directory(
			directory = self.dataset_path+'/train'
			batch_size=32,
			class_mode='categorical')

		val_gen = gen.flow_from_directory(
			directory = self.dataset_path+'/valid'
			traget_size=(224,224),
			batch_size=32,
			class_mode='categorical')

		model = model.compile(
			'Adam',
			loss = tf.keras.losses.CategoricalCrossentropy(),
			metric = ['accuracy']
			)

		model.fit(
			x = train_gen,
			validation_data = val_gen,
			batch_size = 32,
			epochs = 25,
			shuffle = True
			)

		model.save(self.save_path)

	def test():
		pass