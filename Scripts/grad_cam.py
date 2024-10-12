import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def build_simple_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu', name='conv2d')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv2d_1')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def generate_grad_cam(model, img_array, layer_name):
    # Define a função de gradiente para a camada
    layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        layer_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    
    grads = tape.gradient(loss, layer_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    layer_outputs = layer_outputs[0]
    
    # Cria o mapa de calor
    heatmap = layer_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def display_heatmap(heatmap, layer_name):
    plt.matshow(heatmap)
    plt.title(f"Grad-CAM Heatmap for {layer_name}")
    plt.colorbar()
    plt.show()

# Testando o modelo simples
model = build_simple_model()
print("Modelo simples carregado com sucesso")

# Caminho da imagem para análise
img_path = 'C:/Users/User/Desktop/VSCode/SmartAI/data/validation/01_pneumonia/01 (701).jpeg'
img_array = preprocess_image(img_path)

# Testar a camada 'conv2d'
layer_name = 'conv2d'
try:
    print(f"Testando camada: {layer_name}")
    heatmap = generate_grad_cam(model, img_array, layer_name)
    display_heatmap(heatmap, layer_name)
    
except Exception as e:
    print(f"Erro ao gerar Grad-CAM para a camada {layer_name}: {e}")

































