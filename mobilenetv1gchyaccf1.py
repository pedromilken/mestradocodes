import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Definir o tamanho de entrada esperado pela MobileNetV1
input_size = (224, 224)
batch_size = 32

# Caminho para o diretório de dados
data_dir = r'C:\Users\pedro\Downloads\dados'

# Carregar o modelo MobileNetV1 pré-treinado, excluindo a última camada
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Congelar as camadas inferiores para que não sejam treinadas
base_model.trainable = False

# Adicionar camadas finais para ajustar o modelo ao problema de 4 classes
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
])

# Configurar o gerador de imagens com aumento de dados (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

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

# Treinar as camadas finais do modelo
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Agora, descongelar as camadas do MobileNetV1 para fine-tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Extrair características do conjunto de treinamento e validação
features_train = base_model.predict(train_generator)
features_test = base_model.predict(validation_generator)

# Obter os rótulos reais do conjunto de treinamento e validação
Y_train = train_generator.classes
Y_test = validation_generator.classes

# Mapeamento de classes
label_map = {label: idx for label, idx in train_generator.class_indices.items()}

# Treinar o classificador SVM com as características extraídas
classifier = SVC(kernel='rbf', probability=True)
classifier.fit(features_train, Y_train)

# Fazer predições no conjunto de teste e no conjunto de treinamento
Y_pred_test = classifier.predict(features_test)
Y_pred_train = classifier.predict(features_train)

# Calcular e exibir a acurácia e F1-Score para as classes detectadas
accuracy_test = accuracy_score(Y_test, Y_pred_test)
f1_test = f1_score(Y_test, Y_pred_test, average=None)
accuracy_train = accuracy_score(Y_train, Y_pred_train)
f1_train = f1_score(Y_train, Y_pred_train, average=None)

print(f'Acurácia no conjunto de teste: {accuracy_test}')
print(f'Acurácia no conjunto de treinamento: {accuracy_train}')

# Exibir a acurácia para cada classe (GC, H&Y1, H&Y2, H&Y3) no conjunto de teste e treinamento
for i, (label, idx) in enumerate(label_map.items()):
    accuracy_class_test = (Y_test == idx).sum() / len(Y_test)
    accuracy_class_train = (Y_train == idx).sum() / len(Y_train)
    
    print(f'Acurácia para {label} no conjunto de teste: {accuracy_class_test}')
    print(f'Acurácia para {label} no conjunto de treinamento: {accuracy_class_train}')
    
    print(f'F1-Score para {label} no conjunto de teste: {f1_test[i]}')
    print(f'F1-Score para {label} no conjunto de treinamento: {f1_train[i]}')
