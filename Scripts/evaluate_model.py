import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Caminho absoluto para o arquivo do modelo
model_path = 'C:/Users/User/Desktop/VSCode/SmartAI/Models/modelo_patologias_pulmonares.keras'

# Verifique se o arquivo realmente existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File not found: {model_path}")

# Carregar o modelo salvo
model = load_model(model_path)

# Definir o preprocessamento para o conjunto de validação
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/User/Desktop/VSCode/SmartAI/data/validation',  # Caminho absoluto
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Avaliar o desempenho do modelo
loss, accuracy = model.evaluate(validation_generator)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Gerar previsões
predictions = model.predict(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Mapear as classes para os nomes das classes
class_labels = list(validation_generator.class_indices.keys())

# Visualizar algumas previsões
def plot_images(images, labels, predictions, class_labels):
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {class_labels[labels[i]]}\nPred: {class_labels[predictions[i]]}")
        plt.axis('off')
    plt.show()

# Obter as imagens e labels para visualização
image_files = validation_generator.filepaths
labels = validation_generator.classes
images = [plt.imread(img_file) for img_file in image_files]

# Plotar algumas imagens com suas previsões
plot_images(images[:16], labels[:16], predicted_classes[:16], class_labels)


