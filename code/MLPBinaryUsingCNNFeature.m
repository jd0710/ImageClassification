clearvars;

% Load features and labels of training data
load train/train.mat;

%% --browse through the images and look at labels
% for i=1:10
%     clf();
% 
%     % load img
%     img = imread( sprintf('train/imgs/train%05d.jpg', i) );
% 
%     % show img
%     imshow(img);
% 
%     title(sprintf('Label %d', train.y(i)));
% 
%     pause;  % wait for key,Â 
% end

%%split K fold into train/test, use CNN features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

% k-fold!
KFold = 4;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);
train.y = (train.y ~= 4);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

%%
fprintf('Training simple neural network..\n');

addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

rng(5789);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
nn = nnsetup([size(Tr.X,2) 500 250 2]);
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 25;  %  Take a mean gradient step over this many samples
nn.inputZeroMaskedFraction = 0.2;            %  Used for Denoising AutoEncoders
nn.dropoutFraction = 0.2;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
nn.sparsityTarget = 0;         %  Sparsity target
nn.activation_function = 'sigm';

% if == 1 => plots trainin error as the NN is trained
opts.plot = 1;

nn.learningRate = 1;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
Tr.X = Tr.X(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std

% prepare labels for NN
LL = [1*(Tr.y == 0), ...
      1*(Tr.y == 1), ];  % first column, p(y=1)
                        % second column, p(y=2), etc

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
classVote = classVote - ones(size(classVote,1),1);


%USing BER
predErr = BER(classVote,Te.y);

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );


%% visualize samples and their predictions (test set)
% figure;
% for i=20:30  % just 10 of them, though there are thousands
%     clf();
% 
%     img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
%     imshow(img);
% 
% 
%     % show if it is classified as pos or neg, and true label
%     title(sprintf('Label: %d, Pred: %d', train.y(Te.idxs(i)), classVote(i)));
% 
%     pause;  % wait for keydo that then,Â 
% end
