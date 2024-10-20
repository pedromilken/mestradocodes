% Definir o tamanho de entrada esperado pela VGG16
inputSize = [224, 224, 3];

% Importar e separar dados para treinamento e teste
imds = imageDatastore('dados', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
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

% Função auxiliar para calcular precisão, recall e F1-score
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
    
    % Calcular precisão, recall e F1-score
    metrics = calculateMetrics(confMat);
    
    % Exibir métricas
    disp(['Acurácia para a camada ', layer, ': ', num2str(accuracy)]);
    disp(['Precisão média para a camada ', layer, ': ', num2str(nanmean(metrics.precision))]);
    disp(['Recall médio para a camada ', layer, ': ', num2str(nanmean(metrics.recall))]);
    disp(['F1-score médio para a camada ', layer, ': ', num2str(nanmean(metrics.f1))]);
    
    % Calcular a precisão por grupo para o conjunto de treinamento e teste
    classes = unique(imdsTest.Labels);  % Obtém os rótulos únicos
    groupAccuraciesTrain = zeros(length(classes), 1); % Inicializa vetor de acurácia por grupo no treinamento
    groupAccuraciesTest = zeros(length(classes), 1);  % Inicializa vetor de acurácia por grupo no teste
    
    % Previsão para o conjunto de treinamento
    YPredTrain = predict(classifier, featuresTrain);
    
    for j = 1:length(classes)
        class = classes(j);  % Seleciona a classe atual
        
        % Conjunto de Treinamento
        idxTrain = (imdsTrain.Labels == class);
        N1Train = sum(YPredTrain(idxTrain) == YTrain(idxTrain)); % Correções para rótulos e predições
        N2Train = sum(idxTrain);
        if N2Train > 0
            groupAccuraciesTrain(j) = N1Train / N2Train;
        else
            groupAccuraciesTrain(j) = NaN; % Caso não haja exemplos da classe no treinamento
        end
        
        % Conjunto de Teste
        idxTest = (imdsTest.Labels == class);
        N1Test = sum(YPred(idxTest) == YTest(idxTest));
        N2Test = sum(idxTest);
        if N2Test > 0
            groupAccuraciesTest(j) = N1Test / N2Test;
        else
            groupAccuraciesTest(j) = NaN; % Caso não haja exemplos da classe no teste
        end
        
        % Exibe a precisão para a classe atual
        disp(['Acurácia para ', char(class), ' no conjunto de treinamento (camada ', layer, '): ', num2str(groupAccuraciesTrain(j))]);
        disp(['Acurácia para ', char(class), ' no conjunto de teste (camada ', layer, '): ', num2str(groupAccuraciesTest(j))]);
    end
end
