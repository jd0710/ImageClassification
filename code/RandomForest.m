clearvars;

% Load features and labels of training data
load train/train.mat;

%%split half and half into train/test, use CNN features
fprintf('Splitting into train/test..\n');
addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

Tr = [];
Te = [];

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


%when the number of the tree is 100, the testing error: 15.24%
%when the number of the tree is 200, the testing error: 14.63%

%training random forest.
fprintf('Training data..\n');
Forest = TreeBagger(800,Tr.X,Tr.y);

fprintf('Testing data..\n');
classVote = predict(Forest,Te.X);

%Using BER to trainning data
classVote=cellfun(@str2num,classVote);
predErr = BER(classVote,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );