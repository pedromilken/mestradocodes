% Passo 1: Carregar a imagem
image = imread('C:\Users\pedro\OneDrive\Área de Trabalho\ESPIRAIS_NIATS_UFU\CONTROLE\V30GCF0000132ESP1COL12MD.jpg'); % Substitua pelo caminho correto da sua imagem

% Passo 2: Pré-processamento da imagem (convertendo para escala de cinza e binarizando)
grayImage = rgb2gray(image); % Converte para escala de cinza
binaryImage = imbinarize(grayImage, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4); % Binariza a imagem

% Passo 3: Detectar as bordas usando o método de Canny
edges = edge(binaryImage, 'Canny');

% Passo 4: Encontrar as coordenadas dos pixels de borda
[rows, cols] = find(edges);

% Passo 5: Armazenar as coordenadas x e y
x = cols; % Coordenadas x (índices de coluna)
y = rows; % Coordenadas y (índices de linha)

% Passo 6: Criar um arquivo JSON manualmente
fileID = fopen('spiral_coordinates.json', 'w'); % Abrir arquivo para escrita
if fileID == -1
    error('Erro ao abrir o arquivo para escrita.');
end

% Passo 7: Escrever as coordenadas no formato JSON
fprintf(fileID, '{\n'); % Abrir objeto JSON
fprintf(fileID, '    "x": [');

% Escrever coordenadas x
for i = 1:length(x)
    if i == length(x)
        fprintf(fileID, '%d', x(i)); % Último valor, sem vírgula
    else
        fprintf(fileID, '%d, ', x(i)); % Adicionar vírgula para separação
    end
end

fprintf(fileID, '],\n'); % Fechar array x

fprintf(fileID, '    "y": [');

% Escrever coordenadas y
for i = 1:length(y)
    if i == length(y)
        fprintf(fileID, '%d', y(i)); % Último valor, sem vírgula
    else
        fprintf(fileID, '%d, ', y(i)); % Adicionar vírgula para separação
    end
end

fprintf(fileID, ']\n'); % Fechar array y
fprintf(fileID, '}'); % Fechar objeto JSON

fclose(fileID); % Fechar o arquivo

disp('Coordenadas salvas com sucesso no arquivo spiral_coordinates.json');