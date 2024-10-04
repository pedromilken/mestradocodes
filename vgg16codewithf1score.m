% Definir o tamanho de entrada esperado pela VGG16
inputSize = [224, 224, 3];

% Importar e separar dados para treinamento e teste
imds = imageDatastore('dados', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Carregar a rede VGG16
net = vgg16

% Atribuir os rótulos conforme estadiamento e controle
YTrain = zeros(size(imdsTrain.Labels)); % Inicializar rótulos
YTest = zeros(size(imdsTest.Labels));   % Inicializar rótulos

% Atribuir valores de estadiamento e controle
YTrain(imdsTrain.Labels == 'CONTROLE') = 0;
YTrain(imdsTrain.Labels == 'PD') = 1;

YTest(imdsTest.Labels == 'CONTROLE') = 0;
YTest(imdsTest.Labels == 'PD') = 1;
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
% Inicializa as variáveis de Precisão, Recall e F1-Score por grupo para o treinamento e teste
precisionTrain = zeros(length(classes), 1);
recallTrain = zeros(length(classes), 1);
f1scoreTrain = zeros(length(classes), 1);

precisionTest = zeros(length(classes), 1);
recallTest = zeros(length(classes), 1);
f1scoreTest = zeros(length(classes), 1);

% Previsão para o conjunto de treinamento (já foi feito acima, mas repito aqui para clareza)
YPredTrain = predict(classifier, featuresTrain);

for i = 1:length(classes)
    class = classes(i);  % Seleciona a classe atual
    
    % Conjunto de Treinamento
    idxTrain = (imdsTrain.Labels == class);
    
    TPTrain = sum(YPredTrain(idxTrain) == YTrain(idxTrain)); % True Positives (TP)
    FPTrain = sum(YPredTrain(~idxTrain) == YTrain(~idxTrain)); % False Positives (FP)
    FNTrain = sum(YPredTrain(idxTrain) ~= YTrain(idxTrain)); % False Negatives (FN)
    
    % Precisão (Precision)
    if (TPTrain + FPTrain) > 0
        precisionTrain(i) = TPTrain / (TPTrain + FPTrain);
    else
        precisionTrain(i) = NaN;
    end
    
    % Recall (Sensibilidade)
    if (TPTrain + FNTrain) > 0
        recallTrain(i) = TPTrain / (TPTrain + FNTrain);
    else
        recallTrain(i) = NaN;
    end
    
    % F1-Score
    if (precisionTrain(i) + recallTrain(i)) > 0
        f1scoreTrain(i) = 2 * (precisionTrain(i) * recallTrain(i)) / (precisionTrain(i) + recallTrain(i));
    else
        f1scoreTrain(i) = NaN;
    end
    
    % Conjunto de Teste
    idxTest = (imdsTest.Labels == class);
    
    TPTest = sum(YPred(idxTest) == YTest(idxTest)); % True Positives (TP)
    FPTest = sum(YPred(~idxTest) == YTest(~idxTest)); % False Positives (FP)
    FNTest = sum(YPred(idxTest) ~= YTest(idxTest)); % False Negatives (FN)
    
    % Precisão (Precision)
    if (TPTest + FPTest) > 0
        precisionTest(i) = TPTest / (TPTest + FPTest);
    else
        precisionTest(i) = NaN;
    end
    
    % Recall (Sensibilidade)
    if (TPTest + FNTest) > 0
        recallTest(i) = TPTest / (TPTest + FNTest);
    else
        recallTest(i) = NaN;
    end
    
    % F1-Score
    if (precisionTest(i) + recallTest(i)) > 0
        f1scoreTest(i) = 2 * (precisionTest(i) * recallTest(i)) / (precisionTest(i) + recallTest(i));
    else
        f1scoreTest(i) = NaN;
    end
    
    % Exibir os resultados de F1-score para a classe atual
    disp(['F1-Score para ', char(class), ' no conjunto de treinamento: ', num2str(f1scoreTrain(i))]);
    disp(['F1-Score para ', char(class), ' no conjunto de teste: ', num2str(f1scoreTest(i))]);
end
