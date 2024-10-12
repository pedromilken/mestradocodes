% Definir o tamanho de entrada esperado pela VGG16 (ou outra rede)
inputSize = [299, 299, 3];

% Importar e separar dados para treinamento e teste
imds = imageDatastore('dados', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

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

% Definir o objeto de aumento de dados com transformações geométricas
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...         % Aumenta rotação aleatória
    'RandXTranslation', [-10 10], ...      % Aumenta translação horizontal
    'RandYTranslation', [-10 10], ...      % Aumenta translação vertical
    'RandXReflection', true, ...           % Espelhamento horizontal
    'RandScale', [0.9, 1.1], ...           % Zoom aleatório
    'RandXShear', [-10, 10], ...           % Cisalhamento horizontal
    'RandYShear', [-10, 10]);              % Cisalhamento vertical

% Criar um conjunto de imagens aumentado com transformações geométricas
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% Criar um conjunto de teste sem aumento de dados
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Carregar a rede ResNet50 (pode ser trocado por VGG16 ou outra rede)
net = xception();

% Extrair características usando a camada 'avg_pool'
layer = 'block14_sepconv2_act'; % ou outra camada, dependendo da rede
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows');

% Treinar o classificador ECOC (Error-Correcting Output Codes)
classifier = fitcecoc(featuresTrain, YTrain);

% Fazer predições no conjunto de teste
YPred = predict(classifier, featuresTest);

% Calcular a precisão, revocação e F1-Score por classe
classes = unique(YTest);  % Obtém os rótulos únicos
numClasses = length(classes);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    class = classes(i);
    
    % Verdadeiros positivos (TP), falsos positivos (FP) e falsos negativos (FN)
    TP = sum((YPred == class) & (YTest == class));
    FP = sum((YPred == class) & (YTest ~= class));
    FN = sum((YPred ~= class) & (YTest == class));
    
    % Calcular precisão, revocação e F1-Score para a classe atual
    if (TP + FP) > 0
        precision(i) = TP / (TP + FP);
    else
        precision(i) = NaN; % Evita divisão por zero
    end
    
    if (TP + FN) > 0
        recall(i) = TP / (TP + FN);
    else
        recall(i) = NaN; % Evita divisão por zero
    end
    
    if (precision(i) + recall(i)) > 0
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    else
        f1Score(i) = NaN; % Evita divisão por zero
    end
    
    % Exibe os resultados para a classe atual
    disp(['Classe ', num2str(class), ':']);
    disp(['  Precisão: ', num2str(precision(i))]);
    disp(['  Revocação: ', num2str(recall(i))]);
    disp(['  F1-Score: ', num2str(f1Score(i))]);
end

% Calcular o F1-Score médio (macro F1-Score)
macroF1Score = nanmean(f1Score);
disp(['F1-Score médio: ', num2str(macroF1Score)]);
