clearvars;

% Load features and labels of training data
load train/train.mat;
load train/test.mat;


%% split K fold into train/test, use CNN features
fprintf('Splitting into train/test..\n');
%train.X_cnn = train.X_cnn(:,1:end-1);

Tr = [];
Te = [];

% K Fold
KFold = 3;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

%%
fprintf('Training simple neural network..\n');

addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

rng(8339,'twister');  % fix seed, this    NN may be very sensitive to initialization

%% initial nn framework, using one hidden layer containing 90 neurons
nn = nnsetup([size(Tr.X,2) 80 4]);
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
nn.output = 'sigm'; 
nn.learningRate = 2;
nn.scaling_learningRate = 0.9;
nn.momentum = 0.8;
nn.activation_function  = 'sigm'; % activation function

% if == 1 => plots trainin error as the NN is trained
opts.plot = 1;

numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
Tr.X = Tr.X(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

% prepare labels for NN
LL = [1*(Tr.y == 1), ...
      1*(Tr.y == 2), ...
      1*(Tr.y == 3), ...
      1*(Tr.y == 4)];  % first column, p(y=0)
                        % second column, p(y=1), etc

[nn, L] = nntrain(nn, Tr.normX, LL, opts);

Te.normX = normalize(Te.X, mu, sigma);  % normalize test data

% to get the scores we need to do nnff (feed-forward)
%  see for example nnpredict().
% (This is a weird thing of this toolbox)
nn.testing = 1;
nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
nn.testing = 0;


% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~,classVote] = max(nnPred,[],2);

%Using BER
predErr = BER(classVote,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );


%% predit
test.normX = normalize(test.X_cnn, mu, sigma);
nn.testing = 1;
nn = nnff(nn, test.normX, zeros(size(test.normX,1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~,result] = max(nnPred,[],2);

