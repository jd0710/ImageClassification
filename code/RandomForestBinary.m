clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %
% IMPORTANT:
%    Make sure you downloaded the file train.tar.gz provided to you
%    and uncompressed it in the same folder as this file resides.

% Load features and labels of training data
load train/train.mat;

%% -- Example: split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');
addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

Tr = [];
Te = [];
train.y = (train.y ~=4) + ones(length(train.y),1);

KFold = 10;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);

train.X_cnn = double(train.X_cnn);
train.y = double(train.y);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

[Tr.X, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.X = normalize(Te.X, mu, sigma);  % normalize test data


fprintf('Training data..\n');
Forest = TreeBagger(100,Tr.X,Tr.y);

fprintf('Testing data..\n');
classVote = predict(Forest,Te.X);

%Using BER to trainning data
classVote=cellfun(@str2num,classVote);
predErr = BER(classVote,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );