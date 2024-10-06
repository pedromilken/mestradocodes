% Specify the directory containing JSON files
%dataDir = 'C:\Users\pedro\OneDrive\Área de Trabalho\ESPIRAIS_NIATS_UFU\CONTROLE\coordenadascontrole\';
dataDir = 'C:\Users\pedro\OneDrive\Área de Trabalho\ESPIRAIS_NIATS_UFU\PD\estadiamentoH&Y1\coordenadasH&Y1\';
%dataDir = 'C:\Users\pedro\OneDrive\Área de Trabalho\ESPIRAIS_NIATS_UFU\PD\estadiamentoH&Y2\coordenadasH&Y2\';
%dataDir = 'C:\Users\pedro\OneDrive\Área de Trabalho\ESPIRAIS_NIATS_UFU\PD\estadiamentoH&Y3\coordenadasH&Y3';
jsonFiles = dir(fullfile(dataDir, '*.json')); % Get all JSON files in the directory

% Initialize a table to store results
results = table();

% Loop through each JSON file
for k = 1:length(jsonFiles)
    % Construct full file path
    jsonFilePath = fullfile(dataDir, jsonFiles(k).name);
    
    % Load and decode the JSON file with error handling
    try
        fid = fopen(jsonFilePath, 'r');
        raw = fread(fid, inf); % Read the file as bytes
        str = char(raw'); % Convert to a character array
        fclose(fid); % Close the file
        data = jsondecode(str); % Decode JSON
    catch ME
        warning(['Error reading file: ' jsonFiles(k).name ' - ' ME.message]);
        continue; % Skip to the next file if an error occurs
    end

    % Extract x and y coordinates
    x = [data.x];
    y = [data.y];

    % Step 1: Smoothing the data for smoothness and regularity estimation
    windowSize = min(5, length(x) / 10); % Define the size of the smoothing window
    xSmooth = smooth(x, windowSize); % Apply moving average smoothing to x
    ySmooth = smooth(y, windowSize); % Apply moving average smoothing to y

    % Step 2: Calculate curvature (Change in angle between consecutive segments)
    angles = atan2(diff(ySmooth), diff(xSmooth)); % Angles between consecutive points
    curvature = abs(diff(angles)); % Change in angle between consecutive points
    meanCurvature = mean(curvature);
    stdCurvature = std(curvature);

    % Step 3: Estimate the area enclosed by the spiral
    areaEnclosed = polyarea(xSmooth, ySmooth);

    % Step 4: Estimate the perimeter of the spiral (sum of Euclidean distances between points)
    distances = sqrt(diff(xSmooth).^2 + diff(ySmooth).^2); % Euclidean distance between consecutive points
    perimeter = sum(distances);

    % Step 5: Calculate compactness (Perimeter^2 / Area)
    compactness = (perimeter^2) / areaEnclosed;

    % Step 6: Aspect Ratio (Bounding Box Dimensions)
    minX = min(xSmooth);
    maxX = max(xSmooth);
    minY = min(ySmooth);
    maxY = max(ySmooth);
    aspectRatio = (maxX - minX) / (maxY - minY); % Width / Height of the bounding box

    % Step 7: Symmetry (Compare left and right halves of the spiral)
    midX = (minX + maxX) / 2; % Middle of the spiral horizontally
    leftHalfIndices = xSmooth < midX;      % Indices for points to the left of the midline
    rightHalfIndices = xSmooth > midX;     % Indices for points to the right of the midline

    % Extract left and right halves
    leftY = ySmooth(leftHalfIndices);
    rightY = ySmooth(rightHalfIndices);

    % Ensure both halves have the same number of points for comparison
    minLength = min(length(leftY), length(rightY));
    leftY = leftY(1:minLength);
    rightY = rightY(1:minLength);

    % Calculate the vertical distance between left and right points at similar x levels
    horizontalSymmetry = mean(abs(leftY - flip(rightY))); % Flip to compare symmetrically

    % Step 8: Calculate radial deviation (Deviation from the ideal spiral)
    centerX = mean(xSmooth); % Approximate center of the spiral
    centerY = mean(ySmooth); % Approximate center of the spiral

    % Distance from the center for each point
    radii = sqrt((xSmooth - centerX).^2 + (ySmooth - centerY).^2);
    idealRadii = linspace(min(radii), max(radii), length(radii)); % Ideal linear spiral

    % Radial deviation: difference between actual and ideal radii
    radialDeviation = abs(radii - idealRadii');
    meanRadialDeviation = mean(radialDeviation);
    stdRadialDeviation = std(radialDeviation);

    % New Feature Analysis: Proximity Between Points
    % Step 1: Calculate the Euclidean distance between consecutive points
    distances = sqrt(diff(xSmooth).^2 + diff(ySmooth).^2); % Euclidean distance between consecutive points

    % Step 2: Calculate the mean distance (Mean Proximity)
    meanDistance = mean(distances); % Average distance between points

    % Step 3: Calculate standard deviation of distances (variation in proximity)
    stdDistance = std(distances); % Standard deviation of distances

    % Step 4: Define a slowness metric
    % Inversely related to the mean distance: Smaller distances -> Higher slowness
    slownessScore = 1 / (meanDistance + 1e-5); % Adding small constant to avoid division by zero

    % Step 5: Perform FFT on the radial distance to detect tremor frequencies
    L = length(radii); % Length of signal

    if L == 0
        warning('O vetor de raios está vazio. A análise de FFT não pode ser realizada.');
        continue; % Salta para o próximo arquivo, se necessário
    end

    Y = fft(radii); % Fast Fourier Transform of the radial distances
    P2 = abs(Y / L); % Two-sided spectrum
    P1 = P2(1:floor(L/2) + 1); % Single-sided spectrum, ensuring the index is an integer
    P1(2:end-1) = 2 * P1(2:end-1); % Adjust amplitude

    % Frequency domain analysis
    f = (0:(floor(L/2))) / L; % Frequency range

    % Step 6: Identify high-frequency components that may indicate tremor
    highFreqThreshold = 0.1; % Define the threshold for high-frequency content
    highFreqPower = sum(P1(f > highFreqThreshold)); % Sum of high-frequency components

    % Step 7: Tremor Score: Combine high-frequency content and curvature irregularities
    tremorScore = highFreqPower + stdCurvature;

    % Store results in the table
    results = [results; table({jsonFiles(k).name}, areaEnclosed, perimeter, compactness, aspectRatio, ...
                               horizontalSymmetry, meanRadialDeviation, stdRadialDeviation, ...
                               meanCurvature, stdCurvature, meanDistance, stdDistance, ...
                               slownessScore, highFreqPower, tremorScore, ...
                               'VariableNames', {'FileName', 'AreaEnclosed', 'Perimeter', ...
                                                 'Compactness', 'AspectRatio', ...
                                                 'HorizontalSymmetry', 'MeanRadialDeviation', ...
                                                 'StdRadialDeviation', 'MeanCurvature', ...
                                                 'StdCurvature', 'MeanDistance', ...
                                                 'StdDistance', 'SlownessScore', ...
                                                 'HighFreqPower', 'TremorScore'})];

    % Display the extracted features
    fprintf('File: %s\n', jsonFiles(k).name);
    fprintf('Area Enclosed: %.2f\n', areaEnclosed);
    fprintf('Perimeter: %.2f\n', perimeter);
    fprintf('Compactness: %.2f\n', compactness);
    fprintf('Aspect Ratio: %.2f\n', aspectRatio);
    fprintf('Horizontal Symmetry: %.2f\n', horizontalSymmetry);
    fprintf('Mean Radial Deviation: %.2f\n', meanRadialDeviation);
    fprintf('Std Radial Deviation: %.2f\n', stdRadialDeviation);
    fprintf('Mean Curvature: %.2f\n', meanCurvature);
    fprintf('Std Curvature: %.2f\n', stdCurvature);
    fprintf('Mean Distance Between Points (Proximity): %.4f\n', meanDistance);
    fprintf('Standard Deviation of Distances: %.4f\n', stdDistance);
    fprintf('Slowness Score: %.4f\n', slownessScore);
    fprintf('High-Frequency Power (Radial): %.4f\n', highFreqPower);
    fprintf('Tremor Score: %.4f\n\n', tremorScore);
end

% Opção para salvar os resultados em um arquivo Excel
excelFileName = 'spiral_analysis_results1.xlsx';
writetable(results, excelFileName);
fprintf('Os resultados foram salvos em %s\n', excelFileName);

% Abrir o arquivo Excel após a gravação
winopen(excelFileName);
