SmartAI is a project developed in Python, utilizing advanced libraries such as TensorFlow and Keras to build, train, and evaluate convolutional neural network (CNN) models. The objective of the project is to assist in the analysis of chest X-rays, focusing on the detection of pathologies in radiographic images.

Project Features:
Database and Categories:
The training of the model was performed using approximately 5,000 X-ray images, divided into several categories representing different pulmonary conditions: Normal, Pneumonia, Higher density, Lower density, Obstructive diseases, Infectious degenerative diseases, Encapsulated lesions, Mediastinal changes, Thoracic changes.

Classification and Detection of Pathologies:
The model is designed to perform multi-label classification, meaning it can detect multiple diseases in a single image.
Each category was used to train the model to recognize specific patterns, from benign conditions to more complex pathologies.

Detection of Regions of Interest (ROI):
The project incorporates visualization techniques such as Grad-CAM to generate heatmaps, highlighting regions of the image that the model considers relevant for classification.
This helps radiologists and physicians interpret visually and clearly where the model detected anomalies.

Training and Validation Split:
The images were divided into training and validation sets, allowing for effective evaluation and adjustment of the model to ensure better generalization.
The project utilizes metrics such as loss and accuracy to monitor performance during training.

Improvement with Pre-trained Models:
SmartAI also explores the use of pre-trained models, such as DenseNet and CheXNet, fine-tuning them specifically to improve the detection of pulmonary diseases in X-rays.

Project Purpose:
The SmartAI aims to create a clinical decision support system capable of automatically detecting pulmonary anomalies and providing a clear visualization of affected areas. The system has the potential to be validated in clinical environments and eventually integrated into computer-aided diagnosis (CAD) systems, offering valuable insights for physicians and radiologists.
