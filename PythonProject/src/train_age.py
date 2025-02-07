import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.symbolic import as_ge

AUTOTUNE = tf.data.experimental.AUTOTUNE
gender_data_dir = './dataset/age'
test_data_dir = './test/age/'
img_size = (80, 80)
batch_size = 32
age_train_ds, age_val_ds = [
    keras.preprocessing.image_dataset_from_directory(
        gender_data_dir, validation_split=0.2, subset=subset, seed=11,
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

age_class_names = age_train_ds.class_names
age_train_ds = optimize_dataset(age_train_ds, augment=True)
age_val_ds = age_val_ds.cache().prefetch(AUTOTUNE)

#uncomment when training new model

#age_model = keras.models.Sequential([
#        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)),
#        keras.layers.BatchNormalization(),
#        keras.layers.MaxPooling2D((2, 2)),
#
#        keras.layers.Conv2D(64, (3, 3), activation='relu'),
#        keras.layers.BatchNormalization(),
#        keras.layers.MaxPooling2D((2, 2)),
#
#        keras.layers.Conv2D(128, (3, 3), activation='relu'),
#        keras.layers.BatchNormalization(),
#        keras.layers.MaxPooling2D((2, 2)),
#
#        keras.layers.Conv2D(256, (3, 3), activation='relu'),
#        keras.layers.BatchNormalization(),
#        keras.layers.MaxPooling2D((2, 2)),
#
#        keras.layers.Flatten(),
#
#        keras.layers.Dense(512, activation='relu'),
#        keras.layers.Dropout(0.5),
#
#        keras.layers.Dense(256, activation='relu'),
#        keras.layers.Dropout(0.4),
#
#        keras.layers.Dense(len(age_class_names), activation='softmax')
#    ])

#comment when training new model
age_model = keras.models.load_model("agemodel.h5")

optimizer = keras.optimizers.Adam()

age_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

age_model.fit(
    age_train_ds,
    validation_data=age_val_ds,
    epochs=5,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
)

age_model.save('agemodel.h5')

image_batch, label_batch = next(iter(age_val_ds))

img_index = 23
img = image_batch[img_index]

predictions = age_model.predict(tf.expand_dims(img, axis=0))
predicted_class = age_class_names[np.argmax(predictions)]

actual_class = age_class_names[label_batch[img_index]]

plt.imshow(img.numpy().astype("uint8"))
plt.title(f"Predicted: {predicted_class} \nActual: {actual_class}")
plt.axis("off")
plt.show()