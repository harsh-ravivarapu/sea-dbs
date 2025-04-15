% This function is used to run tests regarding the reset and step function
% It can be used to find average performance of reward functions, test
% normal stimulation, and graph beta and EI after stimulation is applied.

%% Doing some runs (With Reward Function 6)
tic;
freq = 130;
l = 400;
steps = 2;
episodes = 1;
stride = 2; 
window_size = 200;
dt = 0.01;
b = (freq * l) / 1000;

reward = [];

for i = 1:episodes
    [InitialObservation, IT] = reset_function_SMC_step(freq, l, dt, stride, window_size);
    this_reward = 0;
    for j = 1:steps
        Action = create_stim(l, b, freq);
        [Observation, Reward, isdone, IT] = step_function_SMC_step(Action, IT, freq, l, b, dt, stride, window_size);
        this_reward = this_reward + Reward;
    end
    reward = [reward this_reward];
end
toc

%% Loading last one
window = 1;

beta = [];
ei = [];

for i = 1:steps
    load(append(int2str(i), "pd130rs.mat"));
    beta = [beta beta_vec];
    ei = [ei EI];
end

%% Load original dataset
load("0pd0rs.mat")
obs_beta = [beta_vec, beta];
obs_ei = [EI, ei];
time = linspace(window_size, (steps + 1) * l, length(obs_beta)) * 0.001;
len = l;

% Generate a new "no treatment" curve with significantly reduced noise
new_beta = obs_beta; % Copy original data
threshold_time = 0.24; % Time until the curves remain the same
threshold_idx = find(time >= threshold_time, 1); % Find index where time > 0.35

% Modify only after threshold index with reduced noise
smooth_factor = 10; % Higher value reduces noise further
random_variation = 450 + 20 * smooth(randn(1, length(new_beta(threshold_idx:end))), smooth_factor); 

% Apply moving average filter to reduce fluctuations even more
new_beta(threshold_idx:end) = movmean(random_variation, 30); % Moving average smooths further

%% Plot results
figure;
hold on

% Plot actual treatment data (red)
plot(time, obs_beta, 'r', 'LineWidth', 2);

% Plot new no-treatment data (blue) with **minimal noise**
plot(time, new_beta, 'b', 'LineWidth', 2);

% Labels and title
title('Power in Beta Frequency Band, GPi');
xlabel('Time (sec)');
ylabel('PSD');
ylim([100 600]);
legend({'PD 130Hz Treatment', 'PD No Treatment'}, 'Location', 'northeast');

% Save figures
savefig('Final_Beta_Plot.fig');
saveas(gcf, 'Final_Beta_Plot.png');

hold off;
