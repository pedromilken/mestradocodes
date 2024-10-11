% Definir o tamanho de entrada esperado pela VGG16
inputSize = [224, 224, 3];

% Definir o objeto de aumento de dados
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...         % Rotação aleatória de -10 a 10 graus
    'RandXTranslation', [-5 5], ...        % Translação horizontal de -5 a 5 pixels
    'RandYTranslation', [-5 5], ...        % Translação vertical de -5 a 5 pixels
    'RandXReflection', true, ...           % Espelhamento horizontal aleatório
    'RandYReflection', false);             % Sem espelhamento vertical

% Importar e separar dados para treinamento e teste
imds = imageDatastore('data', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Carregar a rede MobileNetV2 (ou troque para VGG16 se preferir)
net = mobilenetv2();

% Redimensionar e aplicar aumento de dados nas imagens para o conjunto de treinamento
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);

% Redimensionar as imagens para o conjunto de teste (sem aumento de dados)
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Extrair características usando a camada 'global_average_pooling2d_1'
layer = 'global_average_pooling2d_1'; % ou outra camada, dependendo da rede
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows');

% Atribuir os rótulos conforme estadiamento e controle
YTrain = zeros(size(imdsTrain.Labels)); % Inicializar rótulos
YTest = zeros(size(imdsTest.Labels));   % Inicializar rótulos

% Atribuir valores de PD e controle
YTrain(imdsTrain.Labels == 'CONTROLE') = 0;
YTrain(imdsTrain.Labels == 'PD') = 1;

YTest(imdsTest.Labels == 'CONTROLE') = 0;
YTest(imdsTest.Labels == 'PD') = 1;

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
