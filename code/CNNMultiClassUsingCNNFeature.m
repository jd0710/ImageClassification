clearvars;


% Load features and labels of training data
load train/train.mat;


%% Split half and half into train/test, use CNN features
fprintf('Splitting into train/test..\n');
addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

Tr = [];
Te = [];

% k-fold!
KFold = 5;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);
train.X_cnn = double(train.X_cnn(:,1:36864));
train.y = double(train.y);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);


%normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

Tr.normX = double(reshape(Tr.normX',36,1024,size(Tr.idxs,1)));
Te.normX = double(reshape(Te.normX',36,1024,size(Te.idxs,1)));


%%
fprintf('Training simple neural network..\n');

%rng(5789);

%the construction of CNN 
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 1, 'kernelsize',5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 1, 'kernelsize',3) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};


opts.alpha = 1.5;
opts.plot = 1;
opts.batchsize = 100;
opts.numepochs = 50;


% prepare labels for CNN
LL = [1*(Tr.y == 1), ...
      1*(Tr.y == 2), ...
      1*(Tr.y == 3), ...
      1*(Tr.y == 4)]';  % first column, p(y=0)
                        % second column, p(y=1), etc

cnn = cnnsetup(cnn,Tr.normX,LL);
cnn = cnntrain(cnn, Tr.normX, LL, opts);
cnn = cnnff(cnn, Te.normX);

%plot mean squared error
figure; plot(cnn.rL);

% get the most likely class
[~, classVote] = max(cnn.o);

% use BER
classVote = classVote';
predErr = BER(classVote,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );

