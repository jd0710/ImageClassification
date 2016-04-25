clearvars;

% Load features and labels of training data
load train/train.mat;
load train/test.mat;

%% splitk fold into train/test, use CNN features
fprintf('Splitting into train/test..\n');

addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

Tr = [];
Te = [];

% k-fold!
KFold = 3;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);

train.X_cnn = double(train.X_cnn);
train.y = double(train.y);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);


[Tr.X, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.X = normalize(Te.X, mu, sigma);  % normalize test data


% Setup and train a stacked denoising autoencoder (SAE)
rng(8339,'twister'); 
sae = saesetup([size(Tr.X,2) 512 32]);
%sae = saesetup([size(Tr.X,2) 300]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 2;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =  10;
opts.batchsize = 100;
sae = saetrain(sae, Tr.X, opts);


% Use the SAE to initialize a NN
nn = nnsetup([size(Tr.X,2) 512 32 4]);
%nn = nnsetup([size(Tr.X,2) 300 4]);
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
nn.output = 'sigm'; 
nn.learningRate = 2;
nn.scaling_learningRate = 0.9;
nn.momentum = 0.8;
nn.activation_function  = 'sigm'; % activation function
nn.W{1} = sae.ae{1}.W{1};
%nn.W{2} = sae.ae{2}.W{1};


% prepare labels for NN
LL = [1*(Tr.y == 1), ...
      1*(Tr.y == 2), ...
      1*(Tr.y == 3), ...
      1*(Tr.y == 4)];  % first column, p(y=0)
                        % second column, p(y=1), etc
%train nn
% opts.numepochs =  10;
% opts.batchsize = 50;
opts.plot = 1;
nn = nntrain(nn, Tr.X, LL, opts);

nn.testing = 1;
nn = nnff(nn, Te.X, zeros(size(Te.X,1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~,classVote] = max(nnPred,[],2);

% Using BER
predErr = BER(classVote,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );


test.X_cnn = normalize(test.X_cnn, mu, sigma);  % normalize test data
nn.testing = 1;
nn = nnff(nn, test.X_cnn, zeros(size(test.X_cnn,1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~,Ytest] = max(nnPred,[],2);

