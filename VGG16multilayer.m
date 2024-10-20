% Definir o tamanho de entrada esperado pela VGG16
inputSize = [224, 224, 3];

% Definir o caminho dos dados
dataDir = 'C:\Users\pedro\Downloads\dados'; % Substitua pelo caminho correto

% Importar e separar dados para treinamento e teste
imds = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Carregar a rede VGG16
net = vgg16();

% Redimensionar as imagens para o tamanho de entrada esperado
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Atribuir os rótulos conforme estadiamento e controle
YTrain = zeros(size(imdsTrain.Labels)); % Inicializar rótulos
YTest = zeros(size(imdsTest.Labels));   % Inicializar rótulos

% Atribuir valores de estadiamento e controle
YTrain(imdsTrain.Labels == 'estadiamentoH&Y1') = 1;
YTrain(imdsTrain.Labels == 'estadiamentoH&Y2') = 2;
YTrain(imdsTrain.Labels == 'estadiamentoH&Y3') = 3;
YTrain(imdsTrain.Labels == 'CONTROLE') = 0;

YTest(imdsTest.Labels == 'estadiamentoH&Y1') = 1;
YTest(imdsTest.Labels == 'estadiamentoH&Y2') = 2;
YTest(imdsTest.Labels == 'estadiamentoH&Y3') = 3;
YTest(imdsTest.Labels == 'CONTROLE') = 0;

% Definir as camadas de interesse
layers = {'conv3_3', 'conv5_3', 'fc6', 'fc7'}; % Camadas mais profundas para evitar problemas de memória

% Inicializar células para armazenar as features de cada camada
featuresTrainAllLayers = cell(length(layers), 1);
featuresTestAllLayers = cell(length(layers), 1);

% Função auxiliar para calcular precisão, recall e F1-score por classe
calculateMetrics = @(confMat) struct( ...
    'precision', diag(confMat) ./ sum(confMat, 2), ... % Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)
    'recall', diag(confMat) ./ sum(confMat, 1)', ...  % Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)
    'f1', 2 * diag(confMat) ./ (sum(confMat, 2) + sum(confMat, 1)') ... % 2 * (precisão * recall) / (precisão + recall)
);

% Loop para extrair características de cada camada
for i = 1:length(layers)
    layer = layers{i}; % Seleciona a camada atual
    disp(['Extraindo características da camada: ', layer]);
    
    % Extrair características do conjunto de treinamento com mini-batches
    featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows', 'MiniBatchSize', 32);
    featuresTrainAllLayers{i} = featuresTrain; % Armazena as características extraídas
    
    % Extrair características do conjunto de teste com mini-batches
    featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows', 'MiniBatchSize', 32);
    featuresTestAllLayers{i} = featuresTest; % Armazena as características extraídas
    
    % Treinar o classificador para a camada atual
    classifier = fitcecoc(featuresTrain, YTrain);
    
    % Fazer predições no conjunto de teste
    YPred = predict(classifier, featuresTest);
    
    % Calcular a matriz de confusão
    confMat = confusionmat(YTest, YPred);
    
    % Calcular a acurácia
    accuracy = mean(YPred == YTest);
    
    % Calcular precisão, recall e F1-score para cada classe
    metrics = calculateMetrics(confMat);
    
    % Exibir métricas por grupo
    disp(['Acurácia para a camada ', layer, ': ', num2str(accuracy)]);
    
    % Atribuir os nomes dos grupos para visualização
    groupNames = {'CONTROLE', 'estadiamentoH&Y1', 'estadiamentoH&Y2', 'estadiamentoH&Y3'};
    
    for j = 1:length(groupNames)
        if ~isnan(metrics.f1(j)) % Verifica se o F1-score foi calculado corretamente
            disp(['Precisão para ', groupNames{j}, ' na camada ', layer, ': ', num2str(metrics.precision(j))]);
            disp(['Recall para ', groupNames{j}, ' na camada ', layer, ': ', num2str(metrics.recall(j))]);
            disp(['F1-score para ', groupNames{j}, ' na camada ', layer, ': ', num2str(metrics.f1(j))]);
        else
            disp(['Métricas não disponíveis para ', groupNames{j}, ' na camada ', layer]);
        end
    end
end
