clearvars;

% Load features and labels of training data
load train/train.mat;

%% -- Example: split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');
addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));
addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\piotr_toolbox'));

Tr = [];
Te = [];

KFold = 5;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);

train.X_cnn = double(train.X_cnn);
train.y = double(train.y);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

% max_X = max(Tr.X(:));
% min_X = min(Tr.X(:));
% Tr.X = (Tr.X - min_X)./(max_X- min_X);
% Te.X = (Te.X - min_X)./(max_X- min_X);

[Tr.X, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.X = normalize(Te.X, mu, sigma);  % normalize test data

fprintf('Training data..\n');

%training random forest using another way.
pTrain={'M',100};
forest = forestTrain( Tr.X, Tr.y, pTrain );

fprintf('Testing data..\n');
classVote = forestApply( single(Te.X),forest);

% BER
predErr = BER(classVote,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );