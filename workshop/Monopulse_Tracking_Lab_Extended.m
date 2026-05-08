%% Monopulse Tracking Lab
% This lab is based on an excellent example provided by
%     Dr. Kristin Bell, distinguished Fellow at Metron.  
% 
% In this lab, we use the monopulse array processing technique to measure the 
% off-boresight angle of a transmitter and use these measurements in a Kalman 
% filter to complete a tracking loop.
% 
% Before running this lab, ensure that you have already collected the calibration 
% weights using the "Calibrate Antenna" livescript.

%% Setup
% First, place the HB100 transmitter as close to boresight as possible. For this 
% lab to work, the transmitter has to start at boresight.
% 
% Run the following to clear the workspace.
clear;
close all;
warning('off','MATLAB:system:ObsoleteSystemObjectMixin');

%% Track Gating, Coasting and Prediction Parameters
%
% Adjust these values to explore how each one affects tracking behaviour:
%
% * |SIGNAL_THRESHOLD_DB| — Minimum signal amplitude (dB) required for the
%   Kalman filter to accept a monopulse measurement. Below this level the
%   tracker "coasts": it propagates the prediction but ignores the
%   measurement. Notice how the uncertainty band widens during coasting
%   because there is no measurement to constrain the covariance.
%
% * |MAX_COAST_STEPS| — If the signal stays below threshold for this many
%   consecutive steps the track is declared LOST and the loop exits.
%   This models the track-termination logic in real radar systems.
%
% * |PREDICT_HORIZON_S| — How far ahead (seconds) the dashed prediction
%   trend line extends. Demonstrates how the Kalman velocity state allows
%   short-term extrapolation of target motion.

SIGNAL_THRESHOLD_DB = 70;    % dB — coast if signal falls below this
MAX_COAST_STEPS     = 50;    % consecutive coast steps before LOST
PREDICT_HORIZON_S   = 2.0;   % seconds ahead for prediction trend line

%% 
% Define how long to run the target tracker.

K = 1000;               % max number of measurements to run
tCapture = 60;          % max number of seconds to run
HISTORY_POINTS = 200;   % number of points to keep in plot history

%% 
% Load the calibration weights
load('CalibrationWeights.mat','calibrationweights');

% To prevent from tracking to a sidelobe, we'll only enable elements 4, 5
% elementMask = ones(4,2)  %  Use all 8 elements
elementMask = zeros(4,2); % Use none of the elements
elementMask(4,1) = 1;  % add back in element 4
elementMask(1,2) = 1;  % add back in element 5
calibrationweights.AnalogWeights = calibrationweights.AnalogWeights .* elementMask;

%% 
% Load the frequency of the transmitter being used.
load('HB100_Fc.mat','fc_hb100');

% Collect Monopulse Pattern
% Once the frequency of the HB100 has been identified, we collect the monopulse 
% pattern for the array. With this pattern, we can identify the angle of the transmitter 
% with only a single measurement by looking at the phase different and magnitude 
% ratio between the sum and difference patterns.
% 
% We collect the pattern for 12.5 degrees on either side of boresight. The target 
% azimuth becomes ambiguous outside of this region.
MONOPULSE_VALID_AZ = [-12.5, 12.5];
[monopulsePattern,~] = createMonopulsePattern(fc_hb100,calibrationweights,0,MONOPULSE_VALID_AZ(2));

% Create Antenna Interactor
% The antenna interactor object helps us to interact with the phaser board to 
% steer the beam and collect data. Create this object so that we can control the 
% antenna.
ai = AntennaInteractor(fc_hb100,calibrationweights);

%% Monopulse Tracking
% In this section, we run the monopulse tracking system. 
% 
% First, measure the initial target angle by scanning the receive beam:
angles = -70:70;
startpat = helperGetAmplitude(ai.capturePattern(angles));
close all;
[~,angIdx] = max(abs(startpat));
TRACK_AZ_LIMITS = [min(angles), max(angles)];

%% 
% Setup the initial state. The state consists of the target angle and target 
% angular velocity.
x0 = [angles(angIdx);0];
startAngleInBounds = (x0(1) >= MONOPULSE_VALID_AZ(1)) && (x0(1) <= MONOPULSE_VALID_AZ(2));

%% 
% Setup the initial covariance, we assume that the transmitter is starting within 
% ~10 degrees of boresight and is not moving.
P0 = diag([10;2]);

%% 
% Define the motion model. We make it a function handle so that dt can vary 
% throughout the simulation.
F = @(dt)[1 dt;0 1];

%% 
% Define the motion model covariance.
Q = diag([2^2;5^2]);

%% 
% Define the measurement model. We only measure the target angle, the target 
% angular velocity is not measured directly.
H = [1 0];

%% 
% Define the measurement model variance, which is made up of both the steering 
% error and monopulse error.
R = 2^2;

%%
% Initialize the state and measurement variables.
xhat       = nan(2, K+1);       % Estimated state [angle; angular velocity] at each step
Phat       = zeros(2, 2, K+1);  % Estimated covariance at each step
theta_hat  = nan(1, K);         % Raw monopulse angle measurement at each step
sigAmp_dB  = nan(1, K);         % Signal amplitude history (dB) at each step
isCoasting = false(1, K);       % True at steps where measurement was skipped (coasting)

xhat(:,1)   = x0;
Phat(:,:,1) = P0;

%% Build Display
% The display has two panels:
%
% *Top panel — Angle Tracking:*
%
% * Blue solid line:   Kalman filter track estimate (position state)
% * Blue shaded band:  +/-1 sigma position uncertainty from covariance Phat.
%   Watch this GROW during coasting steps (no measurement to constrain it)
%   and SHRINK when good measurements resume.
% * Green dots:        Raw monopulse angle measurements
% * Red X markers:     Coast steps — measurement was below signal threshold
% * Black dashed line: Forward prediction trend (PREDICT_HORIZON_S ahead)
% * Status label:      Live track state (TRACKING / COASTING / LOST)
%
% *Bottom panel — Signal Quality:*
%
% * Magenta line: Signal amplitude at the steered angle each step (dB)
% * Red dashed:   Coast threshold — drops below this => coast step

fig = figure('Name','Monopulse Tracking Lab','Position',[80 60 1100 720]);

% ---- Top panel: angle tracking ----
ax = subplot(3,1,1:2);
hold(ax,'on'); grid(ax,'on');

% Uncertainty band (drawn first so everything else draws on top)
uncertaintyPatch = fill(ax, [0 0], [0 0], [0.55 0.75 0.95], ...
    'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '\pm1\sigma Uncertainty');

% Prediction trend line (dashed, drawn under main track)
predLine = plot(ax, nan, nan, 'k--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Predicted Trend (+%.0fs)', PREDICT_HORIZON_S));

% Kalman track estimate
angline = plot(ax, nan, nan, 'b-', 'LineWidth', 2, 'DisplayName', 'Track Estimate');

% Raw monopulse measurements
theta_line = plot(ax, nan, nan, 'g.', 'MarkerSize', 9, 'DisplayName', 'Monopulse Measurement');

% Coast markers (red X at each step where signal was below threshold)
coastMarkers = plot(ax, nan, nan, 'rx', 'MarkerSize', 11, 'LineWidth', 2.5, ...
    'DisplayName', 'Coasting (prediction only)');

% Live track-state label
stateText = text(ax, 0.01, 0.97, 'State: INITIALIZING', ...
    'Units', 'normalized', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Color', [0 0.5 0], 'VerticalAlignment', 'top');

title(ax, 'Monopulse Signal Target Tracking');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Angle (degrees)');
ylim(ax, [-70 70]);
legend(ax, 'Location', 'northeast', 'FontSize', 8);

% ---- Bottom panel: signal quality ----
axSig = subplot(3,1,3);
hold(axSig,'on'); grid(axSig,'on');
sigLine = plot(axSig, nan, nan, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Signal Amplitude');
yline(axSig, SIGNAL_THRESHOLD_DB, 'r--', 'LineWidth', 1.5, ...
    'Label', 'Coast Threshold', 'LabelHorizontalAlignment', 'left');
title(axSig, 'Signal Quality');
xlabel(axSig, 'Time (s)');
ylabel(axSig, 'Amplitude (dB)');
legend(axSig, 'Location', 'northeast', 'FontSize', 8);
linkaxes([ax, axSig], 'x');

% If initialization is outside the valid monopulse region, show instruction
% and do not enter the tracking loop.
if ~startAngleInBounds
    msg = sprintf('Place HB100 at boresight and restart script.\nInitial angle %.1f deg is outside valid starting range [%.1f, %.1f] deg.', ...
        x0(1), MONOPULSE_VALID_AZ(1), MONOPULSE_VALID_AZ(2));
    text(ax, 0.5, 0.5, msg, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 24, 'FontWeight', 'bold', 'Color', [0.8 0 0], ...
        'BackgroundColor', [1 1 1], 'Margin', 10);
    set(stateText, 'String', 'State: NOT STARTED', 'Color', [0.8 0 0]);
    drawnow;
    fprintf('\nInitial angle %.1f degrees is outside valid monopulse starting range [%.1f, %.1f] deg.\n', ...
        x0(1), MONOPULSE_VALID_AZ(1), MONOPULSE_VALID_AZ(2));
    fprintf('Place HB100 at boresight and restart script.\n');
    return;
end

% Track the current time for motion updates
t = nan(1, K+1);
t(1)       = 0;
k          = 0;
coastCount = 0;
trackState = "TRACKING";

%% Tracking Loop
% At each iteration the loop:
%
% # *Predicts* the next state using the motion model F (constant velocity).
% # *Measures* signal amplitude at the steered angle to decide whether to gate.
% # If signal >= threshold: *Updates* the Kalman estimate with the monopulse
%   measurement (track tightens).
% # If signal < threshold: *Coasts* — keeps the prediction, skips the update,
%   and increments the coast counter (uncertainty grows).
% # *Terminates* the track if the coast counter exceeds MAX_COAST_STEPS.
% # *Draws* the updated display.

fprintf('\nTrack initialized at %.1f degrees.  Starting tracking loop...\n', x0(1));
tic;
while (toc < tCapture) && (k < K) && (trackState ~= "LOST")
    k = k + 1;

    %--- 1. PREDICT (motion update) -----------------------------------------
    tcurrent = toc;
    dt       = tcurrent - t(k);
    t(k+1)   = tcurrent;
    xm       = F(dt) * xhat(:,k);                      % predicted state
    Pm       = F(dt) * Phat(:,:,k) * F(dt).' + Q;      % predicted covariance
    xm(1)    = clampAzimuth(xm(1), TRACK_AZ_LIMITS);

    %--- 2. MEASURE signal quality at steered angle -------------------------
    steerangle   = clampAzimuth(xm(1), TRACK_AZ_LIMITS);
    sigCapture   = helperGetAmplitude(ai.capturePattern([steerangle]));
    sigAmp_dB(k) = mag2db(max(abs(sigCapture)));

    %--- 3. GATE: accept measurement or coast? ------------------------------
    if sigAmp_dB(k) >= SIGNAL_THRESHOLD_DB
        % Good signal: run the Kalman measurement update
        isCoasting(k) = false;
        coastCount    = 0;

        [sdr, pd]     = ai.captureMonopulsePattern(steerangle);
        oba           = estimateMonopulseAngle(monopulsePattern, sdr, pd);
        theta_hat(k)  = steerangle - oba;

        S             = H * Pm * H.' + R;
        G             = (Pm * H.') / S;         % Kalman gain
        xhat(:,k+1)   = xm + G * (theta_hat(k) - H * xm);
        xhat(1,k+1)   = clampAzimuth(xhat(1,k+1), TRACK_AZ_LIMITS);
        Phat(:,:,k+1) = Pm - G * H * Pm;        % covariance shrinks
    else
        % Weak signal: COAST — trust the prediction, skip the update.
        % The covariance Pm grows because no measurement corrects it.
        isCoasting(k) = true;
        coastCount    = coastCount + 1;
        xhat(:,k+1)   = xm;
        xhat(1,k+1)   = clampAzimuth(xhat(1,k+1), TRACK_AZ_LIMITS);
        Phat(:,:,k+1) = Pm;   % covariance grows — watch the uncertainty band widen!
    end

    %--- 4. TRACK STATE MACHINE ---------------------------------------------
    if coastCount == 0
        trackState = "TRACKING";
    elseif coastCount < MAX_COAST_STEPS
        trackState = "COASTING";
    else
        trackState = "LOST";
    end

    %--- 5. UPDATE DISPLAY --------------------------------------------------
    WIN    = max(k - HISTORY_POINTS, 1);   % rolling window

    % Times and states in the current window
    tWin   = t(WIN:k+1);
    xWin   = xhat(1, WIN:k+1);
    sigWin = sqrt(squeeze(Phat(1, 1, WIN:k+1)))';

    % Uncertainty band (±1σ polygon)
    set(uncertaintyPatch, ...
        'XData', [tWin, fliplr(tWin)], ...
        'YData', [xWin + sigWin, fliplr(xWin - sigWin)]);

    % Kalman track estimate
    set(angline, 'XData', tWin, 'YData', xWin);

    % Raw monopulse measurements (original indexing convention)
    set(theta_line, 'XData', t(WIN:k), 'YData', theta_hat(WIN:k));

    % Coast markers (red X at each coast step)
    cMask    = isCoasting(WIN:k);
    xShifted = xhat(1, WIN+1:k+1);   % post-step estimate aligned to each step
    set(coastMarkers, 'XData', t(WIN:k) .* cMask, ...  % zero-mask unused points
        'YData', xShifted .* cMask);
    % Use NaN instead of zero for suppressed markers so they don't appear at 0
    cxData = t(WIN:k); cxData(~cMask) = nan;
    cyData = xShifted; cyData(~cMask) = nan;
    set(coastMarkers, 'XData', cxData, 'YData', cyData);

    % Forward prediction trend line
    tPred = [t(k+1),      t(k+1) + PREDICT_HORIZON_S];
    xPred = [xhat(1,k+1), xhat(1,k+1) + xhat(2,k+1) * PREDICT_HORIZON_S];
    set(predLine, 'XData', tPred, 'YData', xPred);

    % Signal quality panel
    set(sigLine, 'XData', t(WIN:k), 'YData', sigAmp_dB(WIN:k));

    % Scroll both axes to follow current time
    tNow  = t(k+1);
    tLeft = max(0, tNow - 15);
    xWindow = [tLeft, tNow + PREDICT_HORIZON_S + 0.5];
    xlim(ax, xWindow);

    % Track state label and colour
    switch trackState
        case "TRACKING"
            sCol = [0 0.5 0];
            sMsg = 'State: TRACKING';
        case "COASTING"
            sCol = [0.85 0.5 0];
            sMsg = sprintf('State: COASTING  (%d / %d steps without measurement)', ...
                coastCount, MAX_COAST_STEPS);
        case "LOST"
            sCol = [0.8 0 0];
            sMsg = sprintf('State: TRACK LOST  (%d consecutive coast steps exceeded threshold)', ...
                coastCount);
    end
    set(stateText, 'String', sMsg, 'Color', sCol);

    drawnow;
end

% ---- Post-run summary ------------------------------------------------------
if trackState == "LOST"
    fprintf('\nTrack LOST: %d consecutive steps below %.0f dB signal threshold.\n', ...
        coastCount, SIGNAL_THRESHOLD_DB);
    fprintf('Try reducing MAX_COAST_STEPS or increasing SIGNAL_THRESHOLD_DB to see earlier termination.\n');
else
    fprintf('\nTracking complete: %.1f seconds elapsed, %d measurement steps.\n', toc, k);
    fprintf('Coast steps: %d of %d (%.0f%%)\n', ...
        sum(isCoasting(1:k)), k, 100*sum(isCoasting(1:k))/k);
end

cleanup(ai);
%% Helper Functions

function angle = estimateMonopulseAngle(pattern,sumdiffratio,phasedelta)
    % Return angle of 0 if phase delta is 0
    if phasedelta == 0
        angle = 0;
        return
    end
    
    % Extract monopulse pattern values
    oba = pattern.OBA;
    ampdeltapattern = pattern.SumDiffAmpDelta;
    phasedeltapattern = pattern.SumDiffPhaseDelta;

    % Get the values with the correct phase delta
    validphase = phasedeltapattern == phasedelta;
    validoba = oba(validphase);
    validampdelta = ampdeltapattern(validphase);

    % If the sumdiffratio is outside of bounds, set angle to boundary
    % value. Otherwise interpolate to get angle.
    [maxsdr,maxidx] = max(validampdelta);
    [minsdr,minidx] = min(validampdelta);
    if maxsdr < sumdiffratio
        angle = validoba(maxidx);
    elseif minsdr > sumdiffratio
        angle = validoba(minidx);
    else
        angle = interp1(validampdelta,validoba,sumdiffratio);
    end    
end

function az = clampAzimuth(az, limits)
    % Keep azimuth command in a valid steering range.
    if isnan(az)
        az = mean(limits);
        return
    end
    az = min(max(az, limits(1)), limits(2));
end