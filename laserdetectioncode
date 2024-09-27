% Load the video file using VideoReader
videoFile = 'video4.mp4';
videoReader = VideoReader(videoFile);

% Create a video player for displaying the video
videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

% Initialize an array to store the laser point positions
laserPoints = [];

% Loop over the video frames
while hasFrame(videoReader)
    % Read a video frame
    frame = readFrame(videoReader);
    
    % Convert to HSV to segment the laser point (for red color)
    hsvFrame = rgb2hsv(frame);
    
    % Define thresholds for the red laser in HSV
    % Red exists in two ranges in the hue channel (0-1 scale for Hue)
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
        
        % Append the laser point to the list of tracked points
        laserPoints = [laserPoints; laserPoint]; %#ok<AGROW>
        
        % Mark the detected laser point in the current frame
        frame = insertMarker(frame, laserPoint, 'x', 'Color', 'red', 'Size', 10);
    end
    
    % Display the frame with the tracked laser point
    step(videoPlayer, frame);
end

% Now draw the full trajectory on the final frame
if ~isempty(laserPoints)
    % Use the last frame to draw the entire trajectory
    for i = 2:size(laserPoints, 1)
        % Draw line between consecutive laser points for the complete trajectory
        frame = insertShape(frame, 'Line', [laserPoints(i-1, :) laserPoints(i, :)], ...
                            'Color', 'green', 'LineWidth', 2);
    end
    
    % Save the final frame with the complete trajectory as a JPG image
    imwrite(frame, 'laser_tracking_final_trajectory3.jpg');
    disp('Final image with complete trajectory saved as laser_tracking_final_trajectory.jpg');
end

% Release resources
release(videoPlayer);
