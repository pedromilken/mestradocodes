% Definir o tamanho de entrada esperado pela VGG16
inputSize = [299, 299, 3];

% Importar e separar dados para treinamento e teste
imds = imageDatastore('dados', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Carregar a rede VGG16
net = xception();

% Redimensionar as imagens para o tamanho de entrada esperado
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Extrair características usando a camada 'conv5_3'
layer = 'block14_sepconv2_act';
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows');

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

% Treinar o classificador ECOC (Error-Correcting Output Codes)
classifier = fitcecoc(featuresTrain, YTrain);

% Fazer predições no conjunto de teste
YPred = predict(classifier, featuresTest);

% Calcular a acurácia
accuracy = mean(YPred == YTest);

% Exibir a acurácia
disp(['Acurácia: ', num2str(accuracy)]);

% Calcular a precisão por grupo para o conjunto de treinamento e teste
classes = unique(imdsTest.Labels);  % Obtém os rótulos únicos
groupAccuraciesTrain = zeros(length(classes), 1); % Inicializa vetor de acurácia por grupo no treinamento
groupAccuraciesTest = zeros(length(classes), 1);  % Inicializa vetor de acurácia por grupo no teste

% Previsão para o conjunto de treinamento
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
YPredTrain = predict(classifier, featuresTrain);

for i = 1:length(classes)
    class = classes(i);  % Seleciona a classe atual
    
    % Conjunto de Treinamento
    idxTrain = (imdsTrain.Labels == class);
    N1Train = sum(YPredTrain(idxTrain) == YTrain(idxTrain)); % Correções para rótulos e predições
    N2Train = sum(idxTrain);
    if N2Train > 0
        groupAccuraciesTrain(i) = N1Train / N2Train;
    else
        groupAccuraciesTrain(i) = NaN; % Caso não haja exemplos da classe no treinamento
    end
    
    % Conjunto de Teste
    idxTest = (imdsTest.Labels == class);
    N1Test = sum(YPred(idxTest) == YTest(idxTest));
    N2Test = sum(idxTest);
    if N2Test > 0
        groupAccuraciesTest(i) = N1Test / N2Test;
    else
        groupAccuraciesTest(i) = NaN; % Caso não haja exemplos da classe no teste
    end
    
    % Exibe a precisão para a classe atual
    disp(['Acurácia para ', char(class), ' no conjunto de treinamento: ', num2str(groupAccuraciesTrain(i))]);
    disp(['Acurácia para ', char(class), ' no conjunto de teste: ', num2str(groupAccuraciesTest(i))]);
end
