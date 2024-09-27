% Load the video file using VideoReader
videoFile = 'video4.mp4';
videoReader = VideoReader(videoFile);

% Define the start and end frame numbers
startFrame = 50;  % Start tracking from frame 50
endFrame = 200;   % Stop tracking at frame 200

% Create a video player for displaying the video
videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

% Initialize an array to store the laser point positions and timestamps
laserData = struct('x', {}, 'y', {}, 'time', {});

% Initialize frame counter
frameCount = 0;

% Loop over the video frames
while hasFrame(videoReader)
    % Increment frame counter
    frameCount = frameCount + 1;
    
    % Read a video frame
    frame = readFrame(videoReader);
    
    % Get the timestamp of the current frame
    currentTime = videoReader.CurrentTime;
    
    % Only track between the start and end frames
    if frameCount >= startFrame && frameCount <= endFrame
        % Convert to HSV to segment the laser point (for red color)
        hsvFrame = rgb2hsv(frame);
        
        % Define thresholds for the red laser in HSV
        lowerRed1 = [0, 0.6, 0.6];    % Lower range for red (Hue ~ 0 to 0.05)
        upperRed1 = [0.05, 1.0, 1.0];
        lowerRed2 = [0.95, 0.6, 0.6];  % Upper range for red (Hue ~ 0.95 to 1.0)
        upperRed2 = [1.0, 1.0, 1.0];
        
        % Threshold the HSV image to get the red colors in both ranges
        mask1 = (hsvFrame(:,:,1) >= lowerRed1(1) & hsvFrame(:,:,1) <= upperRed1(1)) & ...
                (hsvFrame(:,:,2) >= lowerRed1(2) & hsvFrame(:,:,2) <= upperRed1(2)) & ...
                (hsvFrame(:,:,3) >= lowerRed1(3) & hsvFrame(:,:,3) <= upperRed1(3));
        
        mask2 = (hsvFrame(:,:,1) >= lowerRed2(1) & hsvFrame(:,:,1) <= upperRed2(1)) & ...
                (hsvFrame(:,:,2) >= lowerRed2(2) & hsvFrame(:,:,2) <= upperRed2(2)) & ...
                (hsvFrame(:,:,3) >= lowerRed2(3) & hsvFrame(:,:,3) <= upperRed2(3));

        % Combine both masks to get all red regions
        mask = mask1 | mask2;

        % Remove small noise by keeping only larger connected components
        mask = bwareaopen(mask, 50); % Remove small objects
        
        % Find the centroid of the detected laser point
        stats = regionprops(mask, 'Centroid');
        
        if ~isempty(stats)
            % Get the centroid of the largest detected area
            centroids = cat(1, stats.Centroid);
            laserPoint = centroids(1, :);
            
            % Append the laser point and timestamp to the list of tracked points
            laserData(end+1).x = laserPoint(1); %#ok<AGROW>
            laserData(end+1).y = laserPoint(2); %#ok<AGROW>
            laserData(end+1).time = currentTime; %#ok<AGROW>
            
            % Mark the detected laser point in the current frame
            frame = insertMarker(frame, laserPoint, 'x', 'Color', 'red', 'Size', 10);
        end
        
        % Display the frame with the tracked laser point
        step(videoPlayer, frame);
    elseif frameCount > endFrame
        % Stop processing after reaching the end frame
        break;
    end
end

% Save the laserData to a JSON file
jsonFileName = 'laser_tracking_data.json';
jsonData = jsonencode(laserData);
fid = fopen(jsonFileName, 'w');
if fid == -1
    error('Cannot create JSON file');
end
fwrite(fid, jsonData, 'char');
fclose(fid);
disp(['Laser tracking data saved as ' jsonFileName]);

% Release resources
release(videoPlayer);

