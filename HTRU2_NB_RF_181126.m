%% -- USING NAIVE BAYES AND RANDOM FOREST CLASSIFIERS TO PREDICT PULSAR OBSERVATIONS IN RADIO EMISSION DATA --
% Using HTRU2 dataset found at the UCI repository: https://archive.ics.uci.edu/ml/datasets/HTRU2
% Stephanie Jury

%% --- 1. DATA IMPORT ---

clc, clear all, close all

rng(1)

filename = '/Users/steffjury/Desktop/Data Science MSc/Machine Learning/Coursework/Pulsar Data/HTRU_2.csv'; %Set custom filepath
delimiter = ',';

% Read columns of data as text
formatSpec = '%s%s%s%s%s%s%s%s%s%[^\n\r]';

% Open text file
fileID = fopen(filename,'r');

% Read columns of data according to the format
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);

% Close the text file
fclose(fileID);

% Convert the contents of columns containing numeric text to numbers and replace non-numeric text with NaN
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8,9]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^[-/+]*\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end

R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Define variables and class columns

HTRU2 = table; % Set type HTRU2 to table

% Define variables
HTRU2.profileMean = cell2mat(raw(:, 1));
HTRU2.profileStd = cell2mat(raw(:, 2));
HTRU2.profileKurt = cell2mat(raw(:, 3));
HTRU2.profileSkew = cell2mat(raw(:, 4));
HTRU2.dmsnrMean = cell2mat(raw(:, 5));
HTRU2.dmsnrStd = cell2mat(raw(:, 6));
HTRU2.dmsnrKurt = cell2mat(raw(:, 7));
HTRU2.dmsnrSkew = cell2mat(raw(:, 8));

% Define class
HTRU2.class = cell2mat(raw(:, 9));

%% Clear temporary variables

clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp R;

%% --- 2. INSPECT DATA AND PREPROCCESS ---

% Inspect table properties
HTRU2_array = table2array(HTRU2);
HTRU2_mean = mean(HTRU2_array);
HTRU2_std = std(HTRU2_array);
HTRU2_skewness = skewness(HTRU2_array);
HTRU2_min = min(HTRU2_array);
HTRU2_max = round(max(HTRU2_array),8);

header = {'Mean','Standard Deviation', 'Skewness','Minimum', 'Maximum'};
HTRU2_summary = [HTRU2_mean' HTRU2_std' HTRU2_skewness' HTRU2_min' HTRU2_max'];
HTRU2_summary_table = [header; num2cell(HTRU2_summary)]

HTRU2_size = size(HTRU2_array)

% Check for missing entries in variable data
idx_t = HTRU2_array(:,1:8)==0;
HRTU2_missing_values =sum(idx_t(:))

% Inspect variable properties
figure(1)
subplot(2,4,1);
hist(HTRU2.profileMean,50);
title('Profile Mean'); 
subplot(2,4,2);
hist(HTRU2.profileStd,50);
title('Profile Std Dev');
subplot(2,4,3);
hist(HTRU2.profileKurt,50);
title('Profile Kurtosis');
subplot(2,4,4);
hist(HTRU2.profileSkew,50);
title('Profile Skew'); 
subplot(2,4,5);
hist(HTRU2.dmsnrMean,50);
title('DM-SNR Mean');
subplot(2,4,6);
hist(HTRU2.dmsnrStd,50);
title('DM-SNR Std Dev');
subplot(2,4,7);
hist(HTRU2.dmsnrKurt,50);
title('DM-SNR Kurtosis') ;
subplot(2,4,8);
hist(HTRU2.dmsnrSkew,50);
title('DM-SNR Skew');
hold off

%% Normalise variable data and replot

% Comment out this section and re-run to test model accuracy without
% normalisation. This does not affect classification error in the NB models
HTRU2{:, 1:8} = normalize(HTRU2{:, 1:8});

figure(2)
subplot(2,4,1);
hist(HTRU2.profileMean,50);
title('Profile Mean');
subplot(2,4,2);
hist(HTRU2.profileStd,50);
title('Profile Std Dev');
subplot(2,4,3);
hist(HTRU2.profileKurt,50);
title('Profile Kurtosis'); 
subplot(2,4,4);
hist(HTRU2.profileSkew,50);
title('Profile Skew');
subplot(2,4,5);
hist(HTRU2.dmsnrMean,50);
title('DM-SNR Mean');
subplot(2,4,6);
hist(HTRU2.dmsnrStd,50);
title('DM-SNR Std Dev');
subplot(2,4,7);
hist(HTRU2.dmsnrKurt,50);
title('DM-SNR Kurtosis');
subplot(2,4,8);
hist(HTRU2.dmsnrSkew,50);
title('DM-SNR Skew') ;

hold off

%% Split data into pulsar and noise categories and inspect distribution by class

pulsar_data = HTRU2(HTRU2.class == 1, :);
noise_data = HTRU2(HTRU2.class == 0, :);

figure(3)
set(gcf,'color','w');
subplot(2,4,1);
plot_pulsar_profileMean = hist(pulsar_data.profileMean,50);
plot(plot_pulsar_profileMean, 'b');
hold on
plot_noise_profileMean = hist(noise_data.profileMean,50);
plot(plot_noise_profileMean, 'r');
title('Profile Mean');

subplot(2,4,2)
plot_pulsar_profileStd = hist(pulsar_data.profileStd,50);
plot(plot_pulsar_profileStd, 'b');
hold on
plot_noise_profileStd = hist(noise_data.profileStd,50);
plot(plot_noise_profileStd, 'r');
title('Profile Std. Dev.')

subplot(2,4,3)
plot_pulsar_profileKurt = hist(pulsar_data.profileKurt,50);
plot(plot_pulsar_profileKurt, 'b');
hold on
plot_noise_profileKurt = hist(noise_data.profileKurt,50);
plot(plot_noise_profileKurt, 'r');
title('Profile Kurtosis'); 

subplot(2,4,4)
plot_pulsar_profileSkew = hist(pulsar_data.profileSkew,50);
plot(plot_pulsar_profileSkew, 'b');
hold on
plot_noise_profileSkew = hist(noise_data.profileSkew,50);
plot(plot_noise_profileSkew, 'r');
title('Profile Skew');

subplot(2,4,5)
plot_pulsar_dmsnrMean = hist(pulsar_data.dmsnrMean,50);
plot(plot_pulsar_dmsnrMean, 'b');
hold on
plot_noise_dmsnrMean = hist(noise_data.dmsnrMean,50);
plot(plot_noise_dmsnrMean, 'r');
title('DM-SNR Mean');

subplot(2,4,6)
plot_pulsar_dmsnrStd = hist(pulsar_data.dmsnrStd,50);
plot(plot_pulsar_dmsnrStd, 'b');
hold on
plot_noise_dmsnrStd = hist(noise_data.dmsnrStd,50);
plot(plot_noise_dmsnrStd, 'r');
title('DM-SNR Std. Dev.');

subplot(2,4,7)
plot_pulsar_dmsnrKurt = hist(pulsar_data.dmsnrKurt,50);
plot(plot_pulsar_dmsnrKurt, 'b');
hold on
plot_noise_dmsnrKurt = hist(noise_data.dmsnrKurt,50);
plot(plot_noise_dmsnrKurt, 'r');
title('DM-SNR Kurtosis') 

subplot(2,4,8)
plot_pulsar_dmsnrSkew = hist(pulsar_data.dmsnrSkew,50);
plot(plot_pulsar_dmsnrSkew, 'b');
hold on
plot_noise_dmsnrSkew = hist(noise_data.dmsnrSkew,50);
plot(plot_noise_dmsnrSkew, 'r');
title('DM-SNR Skew') 

legend('Pulsar','Noise');
hold off

%% Inspect the degree of bias towards the majority class (noise)

pulsar_class = sum(HTRU2.class(:) == 1);
[row, col] = size(HTRU2.class);
HTRU2_bias = pulsar_class/row * 100 % Shows the data is biased towards non-pulsar data

%% Correlation plot and heatmap

% figure(4)
% corr = corrplot(HTRU2)
% title('Dataset Feature and Class Correlation Plot')
% hold off

figure(5)

set(gcf,'color','w');
HTRU2_array = table2array(HTRU2);
corr_matrix = corrcoef(HTRU2_array);
imagesc(corr_matrix); 
set(gca, 'XTick'); 
set(gca, 'YTick');
xticklabels({'Profile Mean', 'Profile Std Dev', 'Profile Kurtosis', 'Profile Skew', 'DM-SNR Mean', 'DM-SNR Std Dev'...
    'DM-SNR Kurtosis', 'DM-SNR Skew', 'Class'});
xtickangle(305);
yticklabels({'Profile Mean', 'Profile Std Dev', 'Profile Kurtosis', 'Profile Skew', 'DM-SNR Mean', 'DM-SNR Std Dev'...
    'DM-SNR Kurtosis', 'DM-SNR Skew', 'Class'});
set(gca, 'YTickLabel');
title('Feature and Class Correlation Matrix');
colormap('hot'); 
colorbar;
hold off

%% Split a hold-out set and training/validation set

[m,n] = size(HTRU2);
p = 0.70; % Puts 70% of the whole dataset into training/validation set and 30% into a hold-out set for testing
idx_t = randperm(m);
training = HTRU2(idx_t(1:round(p*m)),:); 
testing = HTRU2(idx_t(round(p*m)+1:end),:);

% Convert data to array type
training = table2array(training); 
testing = table2array(testing);

%% Remove noise data to make training data less biased towards this class

pulsar_count = sum(training(:,9) == 1); % Sum of pulsar classifications in training set
pulsar_class = training((training(:,9) == 1),:); % Pulsar data only
noise_count = sum(training(:,9) == 0); % Sum of noise classifications in training set
noise_class = training((training(:,9) == 0),:); % Noise data only

% Create a new training set which is evenly distributed between pulsar and noisy data
training_unbiased_ordered = [pulsar_class; noise_class(1:pulsar_count,:)]; 

% Shuffle ordered training data so it can be sliced for cross-validation
training_unbiased = training_unbiased_ordered(randperm(size(training_unbiased_ordered,1)),:);

% Check data loss and remaining training data
data_loss = size((training),1) - size((training_unbiased),1);
size((training_unbiased),1);

%% --- 3. CREATE k-FOLD CROSS VALIDATION SETS ---

k = 10; % Set number of folds
[rows, cols] = size(training_unbiased);

test_slice = round(rows/(k+1)); % Testing data slice size depends on k and length of training data 
% e.g. a testing slice size 200 from a dataset of 2000 rows

training_slice = rows - test_slice; % Training data size is the remainder after test slice removed 
% e.g. remainder is 1800 rows

% Define the first slice
% This is always a special case with test data on top and training data underneath
val_test_1 = training_unbiased(1:test_slice,:);
val_train_1 = training_unbiased(test_slice+1:end,:);

% Initialise variables to store middle slices
val_test = [val_test_1];
val_train = [val_train_1];

for i = 2:k
    % Set test data slice: e.g. take test data from position 201 to position 400 
    new_val_test = training_unbiased(test_slice*(i-1)+1:test_slice*i,:);
    
    % Training data remains above and below the test slice
    % Take training data from position 1 to 200 and concatenate with data
    % from position 401 to the end of the training data
    new_val_train = [training_unbiased(1:test_slice*(i-1),:); training_unbiased(test_slice*i+1:end,:)];
    
    % Add the new cross validation sets to val_test and val_train
    val_test = [val_test ; new_val_test];
    val_train = [val_train ; new_val_train];
end

% Define the final (kth) slice
% This is always a special case with test data on the bottom and training on top
val_test_k = training_unbiased((k-1)*test_slice+1:end,:);
val_train_k = training_unbiased(1:(k-1)*test_slice,:);

% Create vectors which contain all k test and training sets
% Loop through these when applying models
val_test = [val_test ; val_test_k];
val_train = [val_train ; val_train_k];

%% --- 4. DETERMINE WHICH NB MODEL PARAMETERS PERFORM BEST IN VALIDATION

% Initialise variables to store loop outputs 
NB_validation_testslice = [];
NB_cv_class_error = [];
NB_cv_TPR = [];
NB_cv_TNR = [];
NB_cv_PPV = [];
NB_cv_NPV = [];
NB_cv_F1 = [];
NB_cv_runtime = [];

% Set prior distribution 
  prior = [0.5 0.5];
  %prior = [0.6 0.4];
  %prior = [0.7 0.3];
  %prior = [0.8 0.2];
  %prior = [0.9 0.1];

for n = 1:10 % Run the cross validation 10 times

    for i = 1:k

        % Set variables and classes in the validation training slice
        x = val_train(training_slice*i - training_slice + 1:training_slice*i,1:8);
        y = val_train(training_slice*i - training_slice + 1:training_slice*i,9);

        % Time how long the model takes to run
        tic

        % UNCOMMENT THE MODEL TO TRAIN
        % --------------------------------------------------------------------------
        % 1. Naive bayes classifier fitting each predictor using normal distributions
        % ------
        % NB_mdl = fitcnb(x,y,'Prior',prior)
        % ------

        % 2. Naive bayes classifier fitting predictors with skew >=abs(2) using kernel
        % density estimation and others as normal
        % ------
        % NB_mdl = fitcnb(x,y,'Prior',prior,'Distribution',{'normal', 'normal', 'kernel','kernel', 'kernel', 'normal', 'normal', 'kernel'});
        % ------

        % 3. Naive bayes classifier fitting predictors with skew >=abs(1) using kernel desnsity estimation and others as normal
        % ------
         NB_mdl = fitcnb(x,y,'Prior',prior,'Distribution',{'kernel', 'normal', 'kernel', 'kernel', 'kernel', 'kernel', 'normal', 'kernel'});
        % ------

        % 4. Naive bayes classifier fitting each predictor using kernel
        % ------
        % NB_mdl = fitcnb(x,y,'Prior',prior,'Distribution',{'kernel', 'kernel', 'kernel', 'kernel', 'kernel', 'kernel', 'kernel', 'kernel'});
        % ------

        % --------------------------------------------------------------------------
        NB_cv_runtime = [NB_cv_runtime; toc];

        % Select validation test data to pass to model for prediction
        val_slice = val_test(test_slice*i - test_slice + 1:test_slice*i,1:9);

        % Test model on the validation test slice and store posterior probabilities
        [NB_predicted_class, NB_posterior] = predict(NB_mdl,val_slice(:,1:8));

        % Store test slice set number
        NB_validation_testslice = [NB_validation_testslice; i];

        % Store validation set error rates
        true_class = val_slice(:,9);
        correct = sum(true_class == NB_predicted_class);
        NB_cv_class_error_new = 1 - (correct / length(true_class));
        NB_cv_class_error = [NB_cv_class_error ; NB_cv_class_error_new];

        % Calculate error metrics
        [NB_cv_c,NB_cv_order] = confusionmat(true_class,NB_predicted_class);
        NB_cv_TN = NB_cv_c(1,1);
        NB_cv_FN = NB_cv_c(2,1);
        NB_cv_FP = NB_cv_c(1,2);
        NB_cv_TP = NB_cv_c(2,2);
        NB_cv_TPR_new = NB_cv_TP./(NB_cv_TP+NB_cv_FN);
        NB_cv_TNR_new = NB_cv_TN./(NB_cv_TN+NB_cv_FP);
        NB_cv_PPV_new = NB_cv_TP./(NB_cv_TP+NB_cv_FP);
        NB_cv_NPV_new = NB_cv_TN./(NB_cv_TN+NB_cv_FN);
        NB_cv_F1_new = (2*NB_cv_TP)./(2*NB_cv_TP + NB_cv_FP + NB_cv_FN);

        % Store error metrics
        NB_cv_TPR = [NB_cv_TPR; NB_cv_TPR_new];
        NB_cv_TNR = [NB_cv_TNR; NB_cv_TNR_new];
        NB_cv_PPV = [NB_cv_PPV; NB_cv_PPV_new];
        NB_cv_NPV = [NB_cv_NPV; NB_cv_NPV_new];
        NB_cv_F1 = [NB_cv_F1; NB_cv_F1_new];

    end

end

% Inspect NB model validation error
header = {'NB Test Slice Position','Classification Error %','TPR','TNR','PPV','NPV','F1'};
NB_validation_results = [NB_validation_testslice NB_cv_class_error NB_cv_TPR NB_cv_TNR NB_cv_PPV NB_cv_NPV, NB_cv_F1];
NB_validation_stats = [header; num2cell(NB_validation_results)]
NB_mean_cross_validation_error = mean(NB_validation_results(:,2))

% Inspect runtime
NB_cv_runtime_mean = mean(NB_cv_runtime)

%% Tune best model using different kernel widths

% Initialise variables to store loop outputs 
NB_validation_testslice = [];
NB_cv_kernel_class_error = [];
NB_cv_kernel_TPR = [];
NB_cv_kernel_TNR = [];
NB_cv_kernel_PPV = [];
NB_cv_kernel_NPV = [];
NB_cv_kernel_F1 = [];
NB_cv_kernel_runtime = [];

% Fix prior distribution to best observed 
  prior = [0.5 0.5]; 
    
% Set kernal width to test
  %width = 0.01;
  width = 0.1; %Uncomment to produce best tuned model
  %width = 1;
  %width = 5;
  %width = 10;

for n = 1:10 % Run the cross validation 10 times

    for i = 1:k

        % Set variables and classes in the validation training slice
        x = val_train(training_slice*i - training_slice + 1:training_slice*i,1:8);
        y = val_train(training_slice*i - training_slice + 1:training_slice*i,9);

        % Time how long the model takes to run
        tic

        % 3. Naive bayes classifier fitting predictors with skew >=abs(1) using kernel desnsity estimation and others as normal
        % ------
         NB_mdl = fitcnb(x,y,'Prior',prior,'Distribution',...
             {'kernel', 'normal', 'kernel', 'kernel', 'kernel', 'kernel', 'normal', 'kernel'},...
             'Width',width);
        % ------

        NB_cv_kernel_runtime = [NB_cv_kernel_runtime; toc];

        % Select validation test data to pass to model for prediction
        val_slice = val_test(test_slice*i - test_slice + 1:test_slice*i,1:9);

        % Test model on the validation test slice and store posterior probabilities
        [NB_predicted_class, NB_posterior] = predict(NB_mdl,val_slice(:,1:8));

        % Store test slice set number
        NB_validation_testslice = [NB_validation_testslice; i];

        % Store validation set error rates
        true_class = val_slice(:,9);
        correct = sum(true_class == NB_predicted_class);
        NB_cv_kernel_class_error_new = 1 - (correct / length(true_class));
        NB_cv_kernel_class_error = [NB_cv_kernel_class_error ; NB_cv_kernel_class_error_new];

        % Calculate error metrics
        [NB_cv_kernel_c,NB_cv_kernel_order] = confusionmat(true_class,NB_predicted_class);
        NB_cv_kernel_TN = NB_cv_kernel_c(1,1);
        NB_cv_kernel_FN = NB_cv_kernel_c(2,1);
        NB_cv_kernel_FP = NB_cv_kernel_c(1,2);
        NB_cv_kernel_TP = NB_cv_kernel_c(2,2);
        NB_cv_kernel_TPR_new = NB_cv_kernel_TP./(NB_cv_kernel_TP+NB_cv_kernel_FN);
        NB_cv_kernel_TNR_new = NB_cv_kernel_TN./(NB_cv_kernel_TN+NB_cv_kernel_FP);
        NB_cv_kernel_PPV_new = NB_cv_kernel_TP./(NB_cv_kernel_TP+NB_cv_kernel_FP);
        NB_cv_kernel_NPV_new = NB_cv_kernel_TN./(NB_cv_kernel_TN+NB_cv_kernel_FN);
        NB_cv_kernel_F1_new = (2*NB_cv_kernel_TP)./(2*NB_cv_kernel_TP + NB_cv_kernel_FP + NB_cv_kernel_FN);

        % Store error metrics
        NB_cv_kernel_TPR = [NB_cv_kernel_TPR; NB_cv_kernel_TPR_new];
        NB_cv_kernel_TNR = [NB_cv_kernel_TNR; NB_cv_kernel_TNR_new];
        NB_cv_kernel_PPV = [NB_cv_kernel_PPV; NB_cv_kernel_PPV_new];
        NB_cv_kernel_NPV = [NB_cv_kernel_NPV; NB_cv_kernel_NPV_new];
        NB_cv_kernel_F1 = [NB_cv_kernel_F1; NB_cv_kernel_F1_new];

    end

end

%% Inspect best NB model validation error metrics

header = {'NB Test Slice Position','Classification Error %','TPR','TNR','PPV','NPV','F1'};
NB_cv_kernel_results = [NB_validation_testslice NB_cv_kernel_class_error NB_cv_kernel_TPR NB_cv_kernel_TNR...
    NB_cv_kernel_PPV NB_cv_kernel_NPV, NB_cv_kernel_F1];
NB_cv_kernel_stats = [header; num2cell(NB_cv_kernel_results)]
NB_cv_kernel_mean_results = mean(NB_cv_kernel_results(:,2:7))

% Inspect runtime
NB_cv_k_runtime_mean = mean(NB_cv_kernel_runtime)

%% --- 5. DETERMINE WHICH RF MODEL PARAMETERS PERFORM BEST IN VALIDATION

% Set vector of tree numbers to test
% tree_numbers = [1 5 10 20 40 50 100 150 200 250 300]; % Uncomment to produce plots: grown trees vs error and RF model timing plot (5 minutes)
% tree_numbers = [1 5 10 20 40 60 80 100]; % Uncomment to produce surface plot
 tree_numbers = 80; % Uncomment to produce best tuned model (comment out the surface plot code if running one number of grown trees)

% Set vector of minimum leaf sizes to test
% mls = [1 10 20 40 60 80]; % Uncomment to produce surface plot
% mls = 20; % Uncomment to produce plots: grown trees vs error and RF model timing plot
 mls = 1; % Uncomment to produce best tuned model (comment out the surface plot code if running one minimum leaf size)

% Initialise variables to store loop outputs 
    RF_class_error = [];
    RF_cv_TPR = [];
    RF_cv_TNR = [];
    RF_cv_PPV = [];
    RF_cv_NPV = [];
    RF_cv_F1 = [];
    RF_cv_val_slice_error = [];
    RF_cvk_runtime = [];
    RF_cvk_runtime_mean = [];
    
for idx_l = mls
     
    for idx_t = tree_numbers

        RF_cv_k_runtime = [];

        for i = 1:k % Loop through each validation slice

            % Set variables and classes in the validation training slice
            x = val_train(training_slice*i - training_slice + 1:training_slice*i,1:8);
            y = val_train(training_slice*i - training_slice + 1:training_slice*i,9);

            % Time how long the model takes to run
            tic

            RF_mdl = TreeBagger(idx_t,x,y,'OOBVarImp','On','MinLeafSize',idx_l);

            RF_cvk_runtime_new = toc;
            RF_cvk_runtime = [RF_cvk_runtime; RF_cvk_runtime_new];

            % Select validation test data to pass to model for prediction
            val_slice = val_test(test_slice*i - test_slice + 1:test_slice*i,1:9);

            % Test model on the validation test slice and store score probabilities
            [RF_predicted_class, RF_scores] = predict(RF_mdl,val_slice(:,1:8));

            % Convert output array of cells to array
            RF_predicted_class = str2num(cell2mat(RF_predicted_class));

            % Store validation set error rates
            true_class = val_slice(:,9);
            correct = sum(true_class == RF_predicted_class);
            RF_class_error = 1 - (correct / length(true_class));

            % Calculate error metrics
            [RF_cv_c,RF_cv_order] = confusionmat(true_class,RF_predicted_class);
            RF_cv_TN = RF_cv_c(1,1);
            RF_cv_FN = RF_cv_c(2,1);
            RF_cv_FP = RF_cv_c(1,2);
            RF_cv_TP = RF_cv_c(2,2);
            RF_cv_TPR_new = RF_cv_TP./(RF_cv_TP+RF_cv_FN);
            RF_cv_TNR_new = RF_cv_TN./(RF_cv_TN+RF_cv_FP);
            RF_cv_PPV_new = RF_cv_TP./(RF_cv_TP+RF_cv_FP);
            RF_cv_NPV_new = RF_cv_TN./(RF_cv_TN+RF_cv_FN);
            RF_cv_F1_new = (2*RF_cv_TP)./(2*RF_cv_TP + RF_cv_FP + RF_cv_FN);

            % Store error metrics
            RF_cv_TPR = [RF_cv_TPR; RF_cv_TPR_new];
            RF_cv_TNR = [RF_cv_TNR; RF_cv_TNR_new];
            RF_cv_PPV = [RF_cv_PPV; RF_cv_PPV_new];
            RF_cv_NPV = [RF_cv_NPV; RF_cv_NPV_new];
            RF_cv_F1 = [RF_cv_F1; RF_cv_F1_new];

            % Store error metrics by tree and leaf number
            t_vec = idx_t(end,:);
            l_vec = idx_l(end,:);
            t_vec = idx_t(end,:);
            l_vec = idx_l(end,:);
            RF_cv_val_slice_error_new = [t_vec l_vec  i RF_class_error RF_cv_TPR_new...
            RF_cv_TNR_new RF_cv_PPV_new RF_cv_NPV_new RF_cv_F1_new ];
            RF_cv_val_slice_error = [RF_cv_val_slice_error; RF_cv_val_slice_error_new];
        end
        
        RF_cvk_runtime_mean_new = mean(RF_cvk_runtime());
        RF_cvk_runtime_mean = [RF_cvk_runtime_mean; idx_l idx_t RF_cvk_runtime_mean_new];
    end

end

% Inspect tuned model error stats 
header = {'No. Trees','Min. Leaf Size','10-fold CV Slice','Class. Error','TPR','TNR','PPV','NPV','F1'};
RF_validation_stats = [header; num2cell(RF_cv_val_slice_error)];

% Inspect tuned model runtime
header = {'Minimum Leaf Size' 'Trees Grown' 'Mean Runtime in CV'};
RF_leaf_runtime_info = [header; num2cell(RF_cvk_runtime_mean )]
     
%% Loop over cross validation sets to calculate mean error rates per tree number and leaf size

mean_class_error = [];
tree_number = [];
mls_number = [];

for j = k:k:(length(tree_numbers)*length(mls)*k)
    mean_class_error_slice = mean(RF_cv_val_slice_error(j-k+1:j,4:end));
    tree_number = RF_cv_val_slice_error(j,1);
    mls_number = RF_cv_val_slice_error(j,2);
    mean_class_error = [mean_class_error; tree_number mls_number mean_class_error_slice];
end

header = {'No. Trees','Min. Leaf Size','Class. Error','TPR','TNR','PPV','NPV','F1'};
RF_cv_mean_results_stats = [header; num2cell(mean_class_error)]

%% Create surface plot corresponding to minimum classification error with different number of grown trees and minimum leaf size

% Uncomment below only when minimum leaf size and trees grown are uncommented as instructed above

% mesh_x = mean_class_error(:,1);
% mesh_y = mean_class_error(:,2);
% mesh_z = mean_class_error(:,3).*100;
% 
% figure(6)
% set(gcf,'color','w');
% tri = delaunay(mesh_x,mesh_y);
% trisurf(tri,mesh_x,mesh_y,mesh_z);
% hold on 
% shading interp
% colorbar
% alpha 0.8
% xlabel('Number of Grown Trees');
% Bar2Axes = gca();
% ylabel('Minimum Leaf Size');
% set(get(Bar2Axes,'Ylabel'),'Rotation',-25);
% set(get(Bar2Axes,'Xlabel'),'Rotation',15);
% zlabel('Mean Classification Error %');
% set (gca, 'Xdir', 'reverse');
% idxmin = find(mesh_z == min(mesh_z));
% h = scatter3(mesh_x(idxmin),mesh_y(idxmin),mesh_z(idxmin),'filled','r');
% title('Surface Plot of RF Mean Classification Error in Validation')
% h.SizeData = 150;
% legend(h,{['Minimum Mean' newline 'Classification Error']});
% hold off

%% Plot impact of number of trees on mean classification error

%Uncomment below only when minimum leaf size and trees grown are uncommented above

% figure(7)
% set(gcf,'color','w');
% scatter_x = mean_class_error(:,1);
% scatter_y = mean_class_error(:,3)*100;
% scatter(scatter_x, scatter_y);
% hold on
% line_x = linspace(0,300,100);
% line_y = (0.02758.*exp(-0.739.*line_x) + 0.06004.*exp(-0.0000976.*line_x))*100;
% plot(line_x, line_y);
% title('RF Mean Error in Validation vs Number of Grown Trees in Ensemble')
% xlabel('Number of Grown Trees')
% ylabel('Mean Classification Error %')
% hold off

%% Inspect RF model runtime

header = {'Minimum Leaf Size' 'Trees Grown' 'Mean Runtime in CV'};
RF_leaf_runtime_info = [header; num2cell(RF_cvk_runtime_mean )]

%Uncomment below only when minimum leaf size and trees grown are uncommented above

% figure(8)
% runtime_x = RF_cvk_runtime_mean(1:length(tree_numbers),2); % Only plots values for the first minimum leaf size entry
% runtime_y = RF_cvk_runtime_mean(1:length(tree_numbers),3);
% scatter(runtime_x, runtime_y)
% hold on
% line_x = linspace(0,300,100);
% line_y = 0.00286.*line_x; %This line of fit to the runtime points is hardcoded and may not fit the points generated on a different machine
% plot(line_x,line_y);
% xlabel('Number of Grown Trees');
% ylabel('Runtime in Seconds');
% hold off
   
%% Determine feature importance

figure(9)
bar(RF_mdl.OOBPermutedVarDeltaError)
xlabel('Feature Index');
ylabel('Out-of-Bag Feature Importance');
set(gca, 'XTick');
xticklabels({'Profile Mean', 'Profile Std Dev', 'Profile Kurtosis', 'Profile Skew', 'DM-SNR Mean', 'DM-SNR Std Dev'...
    'DM-SNR Kurtosis', 'DM-SNR Skew', 'Class'});
xtickangle(305);
title('RF Feature Importance in Validation');
hold off

%% -- 6. COMPARE BEST NB AND RF MODEL VALIDATION PERFORMANCE

NB_header = {'TPR','TNR', 'PPV', 'NPV','F1','Accuracy'};
NB_cvk_mean_stats = [NB_cv_kernel_mean_results(2:6) 1-NB_cv_kernel_mean_results(1)]
NB_cvk_mean_stats_table = [NB_header; num2cell(NB_cvk_mean_stats)]
% Using:
    % Prior relative frequency distribution of [0.5 0.5]
    % Kernal density estimation where parameter skew is >=1
    % Minimum kernel of 0.1

RF_header = {'TPR','TNR', 'PPV', 'NPV','F1','Accuracy'};
RF_cv_mean_stats = [mean(RF_cv_val_slice_error(:,5:9)) 1-mean(RF_cv_val_slice_error(:,4))];
RF_cv_mean_stats_table = [RF_header; num2cell(RF_cv_mean_stats)]
% Using:
    % Minimum leaf size of 1
    % Number of grown trees 80

cv_bar = [NB_cvk_mean_stats(1) RF_cv_mean_stats(1);...
    NB_cvk_mean_stats(2) RF_cv_mean_stats(2);...
    NB_cvk_mean_stats(3) RF_cv_mean_stats(3);...
    NB_cvk_mean_stats(4) RF_cv_mean_stats(4);...
    NB_cvk_mean_stats(5) RF_cv_mean_stats(5);...
    NB_cvk_mean_stats(6) RF_cv_mean_stats(6)];

x_label = {'TPR','TNR', 'PPV', 'NPV','F1','Accuracy'};
y_label = {'70%','75%','80%','85%','90%','95%','100%'};
y_ticks = linspace(0.7,1,7);

figure(10)
set(gcf,'color','w');
val_bar_chart = bar(cv_bar);
set(gca,'XTickLabel',x_label,'YTick',y_ticks,'YTickLabel',y_label);
ylim([0.7 1]);
title('Best Model Accuracy in Validation');
legend('Naive Bayes', 'Random Forest');
hold off

%% --- 7. APPLY TUNED NB AND RF MODELS TO HELD-OUT TEST DATA ---
    
% NB
    % Test model on the held out test data
    tic
    [NB_test_predicted_class, NB_test_posterior] = predict(NB_mdl,testing(:,1:8));
    NB_test_runtime = toc
    
    % Store test set error rates
    test_true_class = testing(:,9);
    NB_test_correct = sum(test_true_class == NB_test_predicted_class);
    NB_test_class_error = 1 - (NB_test_correct / length(test_true_class));

    % Store true positive, true negative, precision, accuracy and F1 score
    [NB_test_c,NB_test_order] = confusionmat(test_true_class,NB_test_predicted_class);
    
    % Inspect error details averaged over validation sets
    NB_test_TN = NB_test_c(1,1);
    NB_test_FN = NB_test_c(2,1);
    NB_test_FP = NB_test_c(1,2);
    NB_test_TP = NB_test_c(2,2);
    NB_test_TPR = NB_test_TP./(NB_test_TP+NB_test_FN);
    NB_test_TNR = NB_test_TN./(NB_test_TN+NB_test_FP);
    NB_test_PPV = NB_test_TP./(NB_test_TP+NB_test_FP);
    NB_test_NPV = NB_test_TN./(NB_test_TN+NB_test_FN);
    NB_test_F1 = (2*NB_test_TP)./(2*NB_test_TP + NB_test_FP + NB_test_FN);

    header = {'Number TN', 'Number FN', 'Number TP','Number FP','TPR','TNR', 'PPV', 'NPV','F1','Accuracy', 'Mean Class. Error %'};
    NB_test_error_details = [NB_test_TN, NB_test_FN, NB_test_TP, NB_test_FP, NB_test_TPR,...
        NB_test_TNR, NB_test_PPV, NB_test_NPV, NB_test_F1];
    NB_test_error_details_mean = [NB_test_error_details,1-mean(NB_test_class_error), mean(NB_test_class_error)*100];
    NB_test_error_details_table = [header; num2cell(NB_test_error_details_mean)]
    
    
 %RF
    % Test model on the held out test data
    tic
   [RF_test_predicted_class, RF_test_scores] = predict(RF_mdl,testing(:,1:8));
    RF_test_predicted_class = str2num(cell2mat(RF_test_predicted_class));
    RF_test_runtime = toc
    
    % Store test set error rates
    RF_test_correct = sum(test_true_class == RF_test_predicted_class);
    RF_test_class_error = 1 - (RF_test_correct / length(test_true_class));

    % Store true positive, true negative, precision, accuracy and F1 score
    [RF_test_c,RF_test_order] = confusionmat(test_true_class,RF_test_predicted_class);
    
    % Inspect error details averaged over validation sets
    RF_test_TN = RF_test_c(1,1);
    RF_test_FN = RF_test_c(2,1);
    RF_test_FP = RF_test_c(1,2);
    RF_test_TP = RF_test_c(2,2);
    RF_test_TPR = RF_test_TP./(RF_test_TP+RF_test_FN);
    RF_test_TNR = RF_test_TN./(RF_test_TN+RF_test_FP);
    RF_test_PPV = RF_test_TP./(RF_test_TP+RF_test_FP);
    RF_test_NPV = RF_test_TN./(RF_test_TN+RF_test_FN);
    RF_test_F1 = (2*RF_test_TP)./(2*RF_test_TP + RF_test_FP + RF_test_FN);

    header = {'Number TN', 'Number FN', 'Number TP','Number FP','TPR','TNR', 'PPV', 'NPV','F1','Accuracy','Mean Class. Error %'};
    RF_test_error_details = [RF_test_TN, RF_test_FN, RF_test_TP, RF_test_FP, RF_test_TPR,...
        RF_test_TNR, RF_test_PPV, RF_test_NPV, RF_test_F1];
    RF_test_error_details_mean = [RF_test_error_details,1-mean(RF_test_class_error),mean(RF_test_class_error)*100];
    RF_test_error_details_table = [header; num2cell(RF_test_error_details_mean)]
  
%% --- 8. COMPARE BEST NB AND RF MODEL TEST SET PERFORMANCE ---

NB_test_mean_stats = NB_test_error_details_mean(5:10)
RF_test_mean_stats = RF_test_error_details_mean(5:10)

test_bar = [NB_test_mean_stats(1) RF_test_mean_stats(1);...
    NB_test_mean_stats(2) RF_test_mean_stats(2);...
    NB_test_mean_stats(3) RF_test_mean_stats(3);...
    NB_test_mean_stats(4) RF_test_mean_stats(4);...
    NB_test_mean_stats(5) RF_test_mean_stats(5);...
    NB_test_mean_stats(6) RF_test_mean_stats(6)];

x_label = {'TPR','TNR', 'PPV', 'NPV','F1','Accuracy'};
y_label = {'70%','75%','80%','85%','90%','95%','100%'};
y_ticks = linspace(0.7,1,7);

figure(11)
set(gcf,'color','w');
val_bar_chart = bar(test_bar);
set(gca,'XTickLabel',x_label,'YTick',y_ticks,'YTickLabel',y_label);
ylim([0.7 1]);
title('Best Model Accuracy in Testing');
legend('Naive Bayes', 'Random Forest');
hold off

% Plot side by side for comparison with validation performance
cv_test_bar = [NB_cvk_mean_stats(1) RF_cv_mean_stats(1) NB_test_mean_stats(1) RF_test_mean_stats(1);...
    NB_cvk_mean_stats(2) RF_cv_mean_stats(2) NB_test_mean_stats(2) RF_test_mean_stats(2);...
    NB_cvk_mean_stats(3) RF_cv_mean_stats(3) NB_test_mean_stats(3) RF_test_mean_stats(3);...
    NB_cvk_mean_stats(4) RF_cv_mean_stats(4) NB_test_mean_stats(4) RF_test_mean_stats(4);...
    NB_cvk_mean_stats(5) RF_cv_mean_stats(5) NB_test_mean_stats(5) RF_test_mean_stats(5);...
    NB_cvk_mean_stats(6) RF_cv_mean_stats(6) NB_test_mean_stats(6) RF_test_mean_stats(6)];

x_label = {'TPR','TNR', 'PPV', 'NPV','F1','Accuracy'};
y_label = {'70%','75%','80%','85%','90%','95%','100%'};
y_ticks = linspace(0.7,1,7);

figure(12)
set(gcf,'color','w');
val_bar_chart = bar(cv_test_bar);
set(gca,'XTickLabel',x_label,'YTick',y_ticks,'YTickLabel',y_label);
ylim([0.7 1]);
title('Best Model Accuracy in Validation and Testing');
legend('Naive Bayes in Validation', 'Random Forest in Validation', 'Naive Bayes in Testing', 'Random Forest in Testing');
hold off

cv_test_bar = [NB_cvk_mean_stats(1) RF_cv_mean_stats(1) NB_test_mean_stats(1) RF_test_mean_stats(1);...
    NB_cvk_mean_stats(2) RF_cv_mean_stats(2) NB_test_mean_stats(2) RF_test_mean_stats(2);...
    NB_cvk_mean_stats(3) RF_cv_mean_stats(3) NB_test_mean_stats(3) RF_test_mean_stats(3);...
    NB_cvk_mean_stats(4) RF_cv_mean_stats(4) NB_test_mean_stats(4) RF_test_mean_stats(4);...
    NB_cvk_mean_stats(5) RF_cv_mean_stats(5) NB_test_mean_stats(5) RF_test_mean_stats(5);...
    NB_cvk_mean_stats(6) RF_cv_mean_stats(6) NB_test_mean_stats(6) RF_test_mean_stats(6)];

NB_cv_TPR = NB_cvk_mean_stats(1)
RF_cv_TPR = RF_cv_mean_stats(1)
NB_test_TPR = NB_test_mean_stats(1)
NB_test_TPR = RF_test_mean_stats(1)
NB_cv_TNR = NB_cvk_mean_stats(2)
RF_cv_TNR = RF_cv_mean_stats(2)
NB_test_TNR = NB_test_mean_stats(2)
NB_test_TNR = RF_test_mean_stats(2)
NB_cv_PPV = NB_cvk_mean_stats(3)
RF_cv_PPV = RF_cv_mean_stats(3)
NB_test_PPV = NB_test_mean_stats(3)
NB_test_PPV = RF_test_mean_stats(3)
NB_cv_NPV = NB_cvk_mean_stats(4)
RF_cv_NPV = RF_cv_mean_stats(4)
NB_test_NPV = NB_test_mean_stats(4)
NB_test_NPV = RF_test_mean_stats(4)
NB_cv_F1 = NB_cvk_mean_stats(5)
RF_cv_F1 = RF_cv_mean_stats(5)
NB_test_F1 = NB_test_mean_stats(5)
NB_test_F1 = RF_test_mean_stats(5)
NB_cv_acc = NB_cvk_mean_stats(6)
RF_cv_acc = RF_cv_mean_stats(6)
NB_test_acc = NB_test_mean_stats(6)
NB_test_acc = RF_test_mean_stats(6)
 
%% Plot confusion matrices in testing
   
test_true_class_plotconfusion = [test_true_class==0 test_true_class]';

% NB
    figure(13)
    NB_test_posterior_plotconfusion = NB_test_posterior';
    NB_test_conf_plot = plotconfusion(test_true_class_plotconfusion, NB_test_posterior_plotconfusion, 'NB In Testing')
    fh = gcf;
    ah = fh.Children(2);
    ah.XTickLabel{1} = 'Noise';
    ah.XTickLabel{2} = 'Pulsar';
    ah.YTickLabel{1} = 'Noise';
    ah.YTickLabel{2} = 'Pulsar';
    ah.XLabel.String = 'Actual';
    ah.YLabel.String = 'Predicted';
    set(gcf,'color','w');
    hold off
    
    % RF
    figure(14)
    RF_test_scores_plotconfusion = RF_test_scores';
    RF_test_conf_plot = plotconfusion(test_true_class_plotconfusion, RF_test_scores_plotconfusion, 'RF In Testing')
    fh = gcf;
    ah = fh.Children(2);
    ah.XTickLabel{1} = 'Noise';
    ah.XTickLabel{2} = 'Pulsar';
    ah.YTickLabel{1} = 'Noise';
    ah.YTickLabel{2} = 'Pulsar';
    ah.XLabel.String = 'Actual';
    ah.YLabel.String = 'Predicted';
    set(gcf,'color','w');
    hold off

%%  Plot ROC curves for tuned NB and RF model perfomance on held out test data

    % NB
    [NB_X,NB_Y,NB_T,NB_AUC] = perfcurve(test_true_class,NB_test_posterior(:,2),1);
    NB_AUC

    % RF
    [RF_X,RF_Y,RF_T,RF_AUC] = perfcurve(test_true_class,RF_test_scores(:,2),1);
    RF_AUC
    
    % Reference Line
    x_ROC = linspace(0,1,100);
    y_ROC = x_ROC;

    figure(15)
    set(gcf,'color','w');
    plot(x_ROC,y_ROC, '--k', 'HandleVisibility','off')
    hold on
    plot(NB_X,NB_Y)
    plot(RF_X,RF_Y)
    legend('Naive Bayes','Random Forest','Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for NB and RF in Testing')
    hold off
    