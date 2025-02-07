import keras
from keras._tf_keras.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.list_physical_devices('GPU')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.config.run_functions_eagerly(True)
AUTOTUNE = tf.data.experimental.AUTOTUNE
gender_data_dir = './dataset/gender'
img_size = (80, 80)
batch_size = 30
gender_train_ds, gender_val_ds = [
    keras.preprocessing.image_dataset_from_directory(
        gender_data_dir, validation_split=0.4, subset=subset, seed=81,
        image_size=img_size, batch_size=batch_size)
    for subset in ("training", "validation")
]

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    keras.layers.RandomContrast(0.2),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.GaussianNoise(0.1),
])
def optimize_dataset(dataset, augment=True):
    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    return dataset.cache().shuffle(1000).prefetch(AUTOTUNE)

gender_class_names = gender_train_ds.class_names
gender_train_ds = optimize_dataset(gender_train_ds, augment=True)
gender_val_ds = gender_val_ds.cache().prefetch(AUTOTUNE)
weight_decay = 1e-2
input_shape = (img_size[0], img_size[1], 3)
num_classes = len(gender_class_names)
gendermodel = keras.Sequential([
    keras.layers.Rescaling(1.0/255, input_shape=(img_size[0], img_size[1], 3)),

    keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.MaxPooling2D(),


    keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(128, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_classes, activation='softmax')
])

#uncomment when training the existing model
#gendermodel = keras.models.load_model('gendermodel.h5')


optimizer = keras.optimizers.Adam()

gendermodel.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

gendermodel.fit(
    gender_train_ds,
    validation_data=gender_val_ds,
    epochs=5,
    callbacks=[]
)

gendermodel.save('gendermodel.h5')

image_batch, label_batch = next(iter(gender_val_ds))

img_index = 13
img = image_batch[img_index]

predictions = gendermodel.predict(tf.expand_dims(img, axis=0))
predicted_class = gender_class_names[np.argmax(predictions)]

actual_class = gender_class_names[label_batch[img_index]]

plt.imshow(img.numpy().astype("uint8"))
plt.title(f"Predicted: {predicted_class} \nActual: {actual_class}")
plt.axis("off")
plt.show()