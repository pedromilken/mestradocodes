% Carregar o arquivo de vídeo usando VideoReader
videoFile = 'video4.mp4';
videoReader = VideoReader(videoFile);
frameRate = videoReader.FrameRate; % Obter a taxa de frames do vídeo
timePerFrame = 1 / frameRate; % Calcular tempo por frame

% Criar um player de vídeo para exibir o vídeo
videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

% Inicializar um array para armazenar as posições do laser e os tempos correspondentes
laserPoints = [];
trackingData = []; % Array para armazenar tempo, coordenadas x e y

% Loop sobre os frames do vídeo
frameIndex = 0; % Inicializar o índice do frame
while hasFrame(videoReader)
    % Ler um frame do vídeo
    frame = readFrame(videoReader);
    frameIndex = frameIndex + 1; % Incrementar o índice do frame
    
    % Converter para HSV para segmentar o ponto laser (para cor vermelha)
    hsvFrame = rgb2hsv(frame);
    
    % Definir limiares para o laser vermelho em HSV
    lowerRed1 = [0, 0.6, 0.6];
    upperRed1 = [0.05, 1.0, 1.0];
    lowerRed2 = [0.95, 0.6, 0.6];
    upperRed2 = [1.0, 1.0, 1.0];
    
    % Limitar a imagem HSV para obter as cores vermelhas em ambas as faixas
    mask1 = (hsvFrame(:,:,1) >= lowerRed1(1) & hsvFrame(:,:,1) <= upperRed1(1)) & ...
            (hsvFrame(:,:,2) >= lowerRed1(2) & hsvFrame(:,:,2) <= upperRed1(2)) & ...
            (hsvFrame(:,:,3) >= lowerRed1(3) & hsvFrame(:,:,3) <= upperRed1(3));
    
    mask2 = (hsvFrame(:,:,1) >= lowerRed2(1) & hsvFrame(:,:,1) <= upperRed2(1)) & ...
            (hsvFrame(:,:,2) >= lowerRed2(2) & hsvFrame(:,:,2) <= upperRed2(2)) & ...
            (hsvFrame(:,:,3) >= lowerRed2(3) & hsvFrame(:,:,3) <= upperRed2(3));

    % Combinar ambas as máscaras para obter todas as regiões vermelhas
    mask = mask1 | mask2;

    % Remover pequenos ruídos mantendo apenas componentes conectados maiores
    mask = bwareaopen(mask, 50); % Remover pequenos objetos
    
    % Encontrar o centróide do ponto laser detectado
    stats = regionprops(mask, 'Centroid');
    
    if ~isempty(stats)
        % Obter o centróide da maior área detectada
        centroids = cat(1, stats.Centroid);
        laserPoint = centroids(1, :);
        
        % Adicionar o ponto laser à lista de pontos rastreados
        laserPoints = [laserPoints; laserPoint]; %#ok<AGROW>
        
        % Adicionar tempo e coordenadas ao tracking data
        currentTime = frameIndex * timePerFrame; % Calcular tempo atual
        trackingData = [trackingData; currentTime, laserPoint]; %#ok<AGROW>
        
        % Marcar o ponto laser detectado no frame atual
        frame = insertMarker(frame, laserPoint, 'x', 'Color', 'red', 'Size', 10);
    end
    
    % Exibir o frame com o ponto laser rastreado
    step(videoPlayer, frame);
end

% Verifica se há dados de rastreamento
if ~isempty(trackingData)
    % Interpolação para aumentar o número de observações para 50 pontos por segundo
    desiredPointsPerSecond = 50; % Número desejado de pontos por segundo
    totalTime = trackingData(end, 1); % Tempo total da gravação
    interpolatedTimes = linspace(0, totalTime, videoReader.Duration * desiredPointsPerSecond); % Tempos igualmente espaçados
    interpolatedX = interp1(trackingData(:, 1), trackingData(:, 2), interpolatedTimes, 'linear', 'extrap'); % Interpolando X
    interpolatedY = interp1(trackingData(:, 1), trackingData(:, 3), interpolatedTimes, 'linear', 'extrap'); % Interpolando Y
    
    % Criar uma nova matriz de dados de rastreamento com os pontos interpolados
    trackingDataInterpolated = [interpolatedTimes', interpolatedX', interpolatedY'];
    
    % Criar uma estrutura para JSON
    jsonData = struct('time', trackingDataInterpolated(:, 1), 'x', trackingDataInterpolated(:, 2), 'y', trackingDataInterpolated(:, 3));
    jsonString = jsonencode(jsonData); % Converter para formato JSON
    
    % Salvar dados JSON em arquivo
    fid = fopen('laser_tracking_data.json', 'w');
    if fid ~= -1
        fwrite(fid, jsonString, 'char');
        fclose(fid);
        disp('Tracking data saved to laser_tracking_data.json');
    else
        disp('Error saving JSON file.');
    end
    
    % Usar o último frame para desenhar toda a trajetória
    for i = 2:size(laserPoints, 1)
        % Desenhar linha entre pontos consecutivos para a trajetória completa
        frame = insertShape(frame, 'Line', [laserPoints(i-1, :) laserPoints(i, :)], ...
                            'Color', 'green', 'LineWidth', 2);
    end
    
    % Salvar o frame final com a trajetória completa como imagem JPG
    imwrite(frame, 'laser_tracking_final_trajectory.jpg');
    disp('Final image with complete trajectory saved as laser_tracking_final_trajectory.jpg');
end

% Liberar recursos
release(videoPlayer);
