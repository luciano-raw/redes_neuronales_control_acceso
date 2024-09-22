import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Función para mostrar imágenes con sus predicciones
def plot_images(images, true_labels, predicted_labels, class_names, model_name):
    plt.figure(figsize=(15, 7))
    for i in range(min(len(images), 10)):  # Mostrar hasta 10 imágenes
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {class_names[true_labels[i]]}\nPredicted: {class_names[predicted_labels[i]]}')
        plt.axis('off')
    plt.suptitle(f'Visualización de predicciones ({model_name})')
    plt.show()

# Carga y preprocesamiento de datos
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Filtrar solo las clases Automobile (1) y Truck (9)
train_indices = np.where((train_labels == 1) | (train_labels == 9))[0]
test_indices = np.where((test_labels == 1) | (test_labels == 9))[0]

train_images, train_labels = train_images[train_indices], train_labels[train_indices]
test_images, test_labels = test_images[test_indices], test_labels[test_indices]

# Ajustar las etiquetas para que sean 0 o 1
train_labels = np.where(train_labels == 1, 0, 1)
test_labels = np.where(test_labels == 1, 0, 1)

# Convertir etiquetas a categóricas
train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

# Modelado con ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
resnet_model.trainable = False

model_resnet = models.Sequential([
    resnet_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model_resnet.compile(optimizer=Adam(),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Entrenamiento con ResNet50
history_resnet = model_resnet.fit(train_images, train_labels, epochs=20, validation_split=0.2)

# Evaluación con ResNet50
test_loss_resnet, test_acc_resnet = model_resnet.evaluate(test_images, test_labels)
print(f'Accuracy en el conjunto de prueba (ResNet50): {test_acc_resnet}')

# Visualización de métricas y gráficos para ResNet50
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_resnet.history['accuracy'], label='Exactitud')
plt.plot(history_resnet.history['val_accuracy'], label='Exactitud en validación')
plt.xlabel('Epoch')
plt.ylabel('Exactitud')
plt.ylim([0, 1])
plt.legend()
plt.title('Precisión durante el entrenamiento y la validación (ResNet50)')

plt.subplot(1, 2, 2)
plt.plot(history_resnet.history['loss'], label='Pérdida')
plt.plot(history_resnet.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento y la validación (ResNet50)')
plt.show()

# Guardado de modelos
model_resnet.save("modelo_resnet.h5")

# Modelado con VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg_model.trainable = False

model_vgg = models.Sequential([
    vgg_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model_vgg.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Entrenamiento con VGG16
history_vgg = model_vgg.fit(train_images, train_labels, epochs=20, validation_split=0.2)

# Evaluación con VGG16
test_loss_vgg, test_acc_vgg = model_vgg.evaluate(test_images, test_labels)
print(f'Accuracy en el conjunto de prueba (VGG16): {test_acc_vgg}')

# Visualización de métricas y gráficos para VGG16
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_vgg.history['accuracy'], label='Exactitud')
plt.plot(history_vgg.history['val_accuracy'], label='Exactitud en validación')
plt.xlabel('Epoch')
plt.ylabel('Exactitud')
plt.ylim([0, 1])
plt.legend()
plt.title('Precisión durante el entrenamiento y la validación (VGG16)')

plt.subplot(1, 2, 2)
plt.plot(history_vgg.history['loss'], label='Pérdida')
plt.plot(history_vgg.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento y la validación (VGG16)')
plt.show()

# Guardado de modelos
model_vgg.save("modelo_vgg.h5")

# Modelado con CNN
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model_cnn.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Entrenamiento con CNN
history_cnn = model_cnn.fit(train_images, train_labels, epochs=20, validation_split=0.2)

# Evaluación con CNN
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(test_images, test_labels)
print(f'Accuracy en el conjunto de prueba (CNN): {test_acc_cnn}')

# Visualización de métricas y gráficos para CNN
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Exactitud')
plt.plot(history_cnn.history['val_accuracy'], label='Exactitud en validación')
plt.xlabel('Epoch')
plt.ylabel('Exactitud')
plt.ylim([0, 1])
plt.legend()
plt.title('Precisión durante el entrenamiento y la validación (CNN)')

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Pérdida')
plt.plot(history_cnn.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento y la validación (CNN)')
plt.show()

# Guardado de modelos
model_cnn.save("modelo_cnn.h5")

# Imprimir arquitecturas de modelos
print("ResNet50 Model Architecture:")
model_resnet.summary()

print("\nVGG16 Model Architecture:")
model_vgg.summary()

print("\nCNN Model Architecture:")
model_cnn.summary()

# Generación de predicciones y visualización
# Obtener predicciones del conjunto de prueba
predictions_resnet = model_resnet.predict(test_images)
predicted_labels_resnet = np.argmax(predictions_resnet, axis=1)

predictions_vgg = model_vgg.predict(test_images)
predicted_labels_vgg = np.argmax(predictions_vgg, axis=1)

predictions_cnn = model_cnn.predict(test_images)
predicted_labels_cnn = np.argmax(predictions_cnn, axis=1)

# Convertir etiquetas categóricas a etiquetas discretas
true_labels = np.argmax(test_labels, axis=1)

# Mostrar el classification report para ResNet50 
print("Classification Report (ResNet50):")
print(classification_report(true_labels, predicted_labels_resnet, target_names=['Automobile', 'Truck']))

# Mostrar el classification report para VGG16
print("Classification Report (VGG16):")
print(classification_report(true_labels, predicted_labels_vgg, target_names=['Automobile', 'Truck']))

# Mostrar el classification report para CNN
print("Classification Report (CNN):")
print(classification_report(true_labels, predicted_labels_cnn, target_names=['Automobile', 'Truck']))

# Visualizar imágenes con sus predicciones para ResNet50
plot_images(test_images, true_labels, predicted_labels_resnet, ['Automobile', 'Truck'], 'ResNet50')

# Visualizar imágenes con sus predicciones para VGG16
plot_images(test_images, true_labels, predicted_labels_vgg, ['Automobile', 'Truck'], 'VGG16')

# Visualizar imágenes con sus predicciones para CNN
plot_images(test_images, true_labels, predicted_labels_cnn, ['Automobile', 'Truck'], 'CNN')