from tensorflow.keras.models import load_model

# Caminho absoluto para o modelo
model = load_model('C:/Users/User/Desktop/VSCode/SmartAI/models/modelo_patologias_pulmonares.keras')

# Verificar se o modelo foi carregado corretamente
model.summary()

