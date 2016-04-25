clearvars;
load train/train.mat;
load train/test.mat;

%%split K fold into train/test, use CNN features
fprintf('Splitting into train/test..\n');

addpath(genpath('\\files8\data\jwang\My Documents\MATLAB\DeepLearnToolbox-master'));

Tr = [];
Te = [];

% NOTE: you should do this randomly! and k-fold!
train.X_cnn = train.X_cnn(:,1:end-1);
KFold = 4;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

[Tr.X, mu, sigma] = zscore(Tr.X); % train, get mu and std
Te.X = normalize(Te.X, mu, sigma);  % normalize test data


%%
classes = unique(train.y);
SVMModels = cell(size(classes,1),1);
rng(1);

LL = [1*(Tr.y == 1), ...
      1*(Tr.y == 2), ...
      1*(Tr.y == 3), ...
      1*(Tr.y == 4)];  % first column, p(y=0)
                        % second column, p(y=1),

for j = 1:numel(classes);
    indx = LL(:,j); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(Tr.X,indx,'KernelFunction','linear','BoxConstraint',1);
end

N = size(Te.X,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},Te.X);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

predErr = BER( maxScore,Te.y );

fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );

test.X_cnn = test.X_cnn(:,1:end-1);
test.X_cnn = normalize(test.X_cnn, mu, sigma);  % normalize test data
N = size(test.X_cnn,1);
YScores = zeros(N,numel(classes));

for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},test.X_cnn);
    YScores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,Ytest] = max(YScores,[],2);

