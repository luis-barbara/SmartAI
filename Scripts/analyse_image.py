import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Caminho para o arquivo do modelo
model_path = 'C:/Users/User/Desktop/VSCode/SmartAI/Models/modelo_patologias_pulmonares.keras'

# Carregar o modelo
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# Função para carregar e preprocessar a imagem
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalização
    return img_array

# Função para classificar a imagem
def classify_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['Normal', 'Pneumonia', 'Maior Densidade', 'Menor Densidade', 
                    'Obstrutivas', 'Infeciosas Degenerativas', 'Lesões Encapsuladas', 
                    'Alterações Mediastino', 'Alterações Tórax']
    predicted_class = class_labels[class_idx]
    confidence = predictions[0][class_idx]
    return predicted_class, confidence

# Analisar a imagem
img_path = 'C:/Users/User/Desktop/VSCode/SmartAI/data/test/02_maior_densidade/00 (1).jpg'
predicted_class, confidence = classify_image(img_path)
print(f'Classe Predita: {predicted_class}')
print(f'Confiança: {confidence:.2f}')

# Visualizar a imagem
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.title(f'Classe Predita: {predicted_class}\nConfiança: {confidence:.2f}')
plt.show()

