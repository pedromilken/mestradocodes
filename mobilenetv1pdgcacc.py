import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Definir o tamanho de entrada esperado pela MobileNetV1
input_size = (224, 224)
batch_size = 32

# Caminho para o diretório de dados
data_dir = r'C:\Users\pedro\Downloads\data'

# Carregar o modelo MobileNetV1 pré-treinado, excluindo a última camada
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Configurar o gerador de imagens
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.3)

# Preparar dados de treinamento e validação
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Extrair características do conjunto de treinamento
features_train = base_model.predict(train_generator)
Y_train = train_generator.classes

# Extrair características do conjunto de teste
features_test = base_model.predict(validation_generator)
Y_test = validation_generator.classes

# Treinar o classificador SVM
classifier = SVC(kernel='linear', probability=True)
classifier.fit(features_train, Y_train)

# Fazer predições no conjunto de teste
Y_pred = classifier.predict(features_test)

# Calcular a acurácia
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Acurácia: {accuracy}')

# Calcular a acurácia por grupo (Controle e PD)
unique_classes = np.unique(Y_test)
group_accuracies_train = []
group_accuracies_test = []

# Previsão para o conjunto de treinamento
Y_pred_train = classifier.predict(features_train)

for class_label in unique_classes:
    # Conjunto de Treinamento
    idx_train = (Y_train == class_label)
    accuracy_train = np.mean(Y_pred_train[idx_train] == Y_train[idx_train]) if np.sum(idx_train) > 0 else np.nan
    group_accuracies_train.append(accuracy_train)
    
    # Conjunto de Teste
    idx_test = (Y_test == class_label)
    accuracy_test = np.mean(Y_pred[idx_test] == Y_test[idx_test]) if np.sum(idx_test) > 0 else np.nan
    group_accuracies_test.append(accuracy_test)
    
    # Exibir a precisão para a classe atual
    class_name = 'CONTROLE' if class_label == 0 else 'PD'
    print(f'Acurácia para {class_name} no conjunto de treinamento: {accuracy_train}')
    print(f'Acurácia para {class_name} no conjunto de teste: {accuracy_test}')
