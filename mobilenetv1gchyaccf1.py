import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# Definir o tamanho de entrada esperado pela MobileNetV1
input_size = (224, 224)
batch_size = 32

# Caminho para o diretório de dados
data_dir = r'C:\Users\pedro\Downloads\dados'
# Carregar o modelo MobileNetV1 pré-treinado, excluindo a última camada
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Configurar o gerador de imagens
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.3)

# Preparar dados de treinamento e validação
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Extrair características do conjunto de treinamento
features_train = base_model.predict(train_generator)
Y_train = train_generator.classes

# Extrair características do conjunto de teste
features_test = base_model.predict(validation_generator)
Y_test = validation_generator.classes

# Atribuir valores numéricos para cada classe:
# CONTROLE = 0, estadiamentoH&Y1 = 1, estadiamentoH&Y2 = 2, estadiamentoH&Y3 = 3
label_map = {'CONTROLE': 0, 'estadiamentoH&Y1': 1, 'estadiamentoH&Y2': 2, 'estadiamentoH&Y3': 3}
Y_train = np.vectorize(lambda x: label_map[train_generator.class_indices[x]])(Y_train)
Y_test = np.vectorize(lambda x: label_map[validation_generator.class_indices[x]])(Y_test)

# Treinar o classificador SVM
classifier = SVC(kernel='linear', probability=True)
classifier.fit(features_train, Y_train)

# Fazer predições no conjunto de teste
Y_pred = classifier.predict(features_test)

# Calcular a acurácia geral
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Acurácia geral no conjunto de teste: {accuracy}')

# Calcular F1-Score por classe para o conjunto de teste
f1_test = f1_score(Y_test, Y_pred, average=None, labels=np.unique(Y_test))
for i, label in enumerate(['CONTROLE', 'estadiamentoH&Y1', 'estadiamentoH&Y2', 'estadiamentoH&Y3']):
    print(f'F1-Score para {label} no conjunto de teste: {f1_test[i]}')

# Calcular acurácia e F1-Score por grupo para o conjunto de treinamento
Y_pred_train = classifier.predict(features_train)

# Acurácia e F1-Score por classe no conjunto de treinamento
accuracy_train = accuracy_score(Y_train, Y_pred_train)
f1_train = f1_score(Y_train, Y_pred_train, average=None, labels=np.unique(Y_train))
print(f'Acurácia geral no conjunto de treinamento: {accuracy_train}')
for i, label in enumerate(['CONTROLE', 'estadiamentoH&Y1', 'estadiamentoH&Y2', 'estadiamentoH&Y3']):
    print(f'F1-Score para {label} no conjunto de treinamento: {f1_train[i]}')