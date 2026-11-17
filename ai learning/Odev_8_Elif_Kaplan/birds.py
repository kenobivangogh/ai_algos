import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np

#0,1,2,3,4 sınıfları kullanıldı

def split_dataset(base_dir, output_dir, test_size=100, val_size=50):
    """
    Veri setini train, validation ve test olarak böler.
    base_dir: Ana veri seti klasörü (train klasörü)
    output_dir: Bölünmüş veri setinin kaydedileceği ana klasör
    test_size: Test seti için ayrılacak resim sayısı
    val_size: Validation seti için ayrılacak resim sayısı
    """

    split_folders = ['train', 'validation', 'test']
    for folder in split_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)


    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)

        if os.path.isdir(class_dir):  
            images = os.listdir(class_dir)
            images.sort()
            test_images = images[:test_size]
            val_images = images[test_size:test_size + val_size]
            train_images = images[test_size + val_size:]

            for img in test_images:
                shutil.copy(os.path.join(class_dir, img), 
                            os.path.join(output_dir, 'test', class_name + '_' + img))

            for img in val_images:
                shutil.copy(os.path.join(class_dir, img), 
                            os.path.join(output_dir, 'validation', class_name + '_' + img))

            for img in train_images:
                shutil.copy(os.path.join(class_dir, img), 
                            os.path.join(output_dir, 'train', class_name + '_' + img))

    print("Veri seti train, validation ve test olarak bölündü")


base_directory = 'Birds/train'  
output_directory = 'deneme'  

#split_dataset(base_directory, output_directory)

train_dir = 'deneme/train'      
validation_dir = 'deneme/validation'  
test_dir = 'deneme/test'  

train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    #os.path.join(output_directory, 'train'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    #os.path.join(output_directory, 'validation'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    #os.path.join(output_directory, 'test'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#  CNN Modeli Oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=1
)

# Doğruluk
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
