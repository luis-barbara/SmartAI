import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

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
    
    heatmap = layer_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def overlay_heatmap_on_image(heatmap, img_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    img = np.float32(img) / 255
    overlay = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return overlay

def display_image_with_heatmap(overlay):
    plt.imshow(overlay)
    plt.title("Overlay of Grad-CAM Heatmap")
    plt.axis('off')
    plt.show()

def highlight_regions_of_interest(heatmap, img_path, threshold_factor=0.1, min_contour_area=500):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    
    # Suavizar o heatmap com filtro mediano
    heatmap = cv2.medianBlur(heatmap, 5)
    
    # Aplicar limiar dinâmico para binarização
    threshold_value = threshold_factor * np.max(heatmap)
    _, binary_heatmap = cv2.threshold(heatmap, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Aplicar operações morfológicas para melhorar a detecção
    kernel = np.ones((15, 15), np.uint8)
    binary_heatmap = cv2.dilate(binary_heatmap, kernel, iterations=2)
    binary_heatmap = cv2.erode(binary_heatmap, kernel, iterations=1)
    
    # Encontrar contornos a partir do heatmap binarizado
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img

# Testando o modelo simples
model = build_simple_model()
print("Modelo simples carregado com sucesso")

# Caminho da imagem para análise
img_path = 'C:/Users/User/Desktop/VSCode/SmartAI/data/train/02_maior_densidade/02 (1).jpeg'
img_array = preprocess_image(img_path)

# Testar a camada 'conv2d'
layer_name = 'conv2d'
try:
    print(f"Testando camada: {layer_name}")
    heatmap = generate_grad_cam(model, img_array, layer_name)
    
    # Sobrepor o heatmap na imagem original
    overlay = overlay_heatmap_on_image(heatmap, img_path)
    display_image_with_heatmap(overlay)
    
    # Destacar regiões de interesse com base no heatmap binarizado
    highlighted_image = highlight_regions_of_interest(heatmap, img_path, threshold_factor=0.1, min_contour_area=500)
    plt.imshow(highlighted_image)
    plt.title("Highlighted Regions of Interest")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Erro ao gerar Grad-CAM para a camada {layer_name}: {e}")













