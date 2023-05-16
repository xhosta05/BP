import tensorflow as tf;
from tensorflow.keras import layers

#    ValueError: Shapes (2, 1080, 1920) and (2, 1031, 1871, 3) are incompatible
def dummy_model(input_shape):
	model = tf.keras.Sequential([
		layers.Conv2D(3, 50, activation='relu',  input_shape=input_shape),
		layers.Conv2DTranspose(5, 50, activation='relu'),
		layers.Dense(1, activation=tf.keras.activations.softmax)
		])
	# model.save("model.keras")
	return model



