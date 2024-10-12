import tensorflow as tf
from tensorflow.keras.models import load_model

# Carregar o modelo salvo
model_path = 'C:/Users/User/Desktop/VSCode/SmartAI/Models/modelo_patologias_pulmonares.keras'
model = load_model(model_path)

# Listar todas as camadas do modelo
print("Camadas do modelo:")
for layer in model.layers:
    print(layer.name, layer.__class__.__name__)

