import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Definindo o modelo CNN para detecção de patologias pulmonares
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')  # Para múltiplas classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preprocessamento das imagens de treinamento
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=10, 
    zoom_range=0.1, 
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    '../data/train',  # Caminho para as imagens de treino
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'  # Usar sparse para labels inteiros
)

# Preprocessamento das imagens de validação
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    '../data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'  # Usar sparse para labels inteiros
)

# Checkpoints para salvar o melhor modelo
checkpoint = ModelCheckpoint(
    '../models/modelo_patologias_pulmonares.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Treinamento do modelo
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Visualizar o progresso do treinamento
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
