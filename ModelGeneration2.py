#https://www.tensorflow.org/tutorials/images/classification
#https://www.edureka.co/blog/tensorflow-image-classification
#https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
#https://cv-tricks.com/tensorflow-tutorial/nsfw-tensorflow-identifying-objectionable-content-deep-learning/
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2


data_dir = 'backEnd/nsfw_detector/customDetector/training_full'

batch_size = 32
img_height = 512
img_width = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  #color_mode="grayscale"
  )

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  #color_mode="grayscale"
  )

class_names = train_ds.class_names
print("Class Names")
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
  ]
)

activation_function = "relu"
epochs = 5
opt_1 = Adam(learning_rate=0.001)

model = Sequential([
  layers.Input(shape=(512,512,3)),
  #layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 4, padding='same', activation=activation_function),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 4, padding='same', activation=activation_function),
  layers.MaxPooling2D(),
  #layers.Dropout(0.02),
  layers.Conv2D(64, 4, padding='same', activation=activation_function),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation=activation_function),
  layers.Dense(num_classes)
])

model.compile(optimizer=opt_1,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#save model
model.save("./backEnd/nsfw_detector/customDetector/myModels/myModel_small.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

