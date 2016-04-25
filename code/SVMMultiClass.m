clearvars;
load train/train.mat;
load train/test.mat;

%%split K fold into train/test, use CNN features
fprintf('Splitting into train/test..\n');

addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

Tr = [];
Te = [];

% k-fold!
train.X_cnn = train.X_cnn(:,1:end-1);
KFold = 5;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

[Tr.X, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.X = normalize(Te.X, mu, sigma);  % normalize test data

%%
fprintf('Training data with SVM..\n');

%Error 9.39%
%Mdl = fitcecoc(Tr.X,Tr.y);
%label = predict(Mdl,Te.X);

%'rbf' Gaussian Kernel 
%'polynomial' Polynomial Kernel 
%'linear' Kernel

% fun = templateSVM('Standardize',1,'KernelFunction','linear');
% Mdl = fitcecoc(Tr.X,Tr.y,'Learners',fun);

Mdl = fitcecoc(Tr.X,Tr.y);
label = predict(Mdl,Te.X);

predErr = BER( label,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );

test.X_cnn = test.X_cnn(:,1:end-1);
test.X_cnn = normalize(test.X_cnn, mu, sigma);  % normalize test data
Ytest = predict(Mdl,test.X_cnn);
