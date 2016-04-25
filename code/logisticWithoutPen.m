clear;
load train/train.mat;
x = train.X_cnn;
y = (train.y~=4);

[~, minIndex] = min(x(:,28));
x(minIndex, :) = [];
y(minIndex) = [];
[~, minIndex] = min(x(:,3));
x(minIndex, :) = [];
y(minIndex) = [];
[~, minIndex] = min(x(:,12));
x(minIndex, :) = [];
y(minIndex) = [];
[~, maxIndex] = max(x(:,28));
x(maxIndex, :) = [];
y(maxIndex) = [];

N = length(y);

%K fold cross validation
K = 5;
propotion = 1 - 1./K;
degree = 1:2;
zeroOneLossTe = zeros(length(K),length(degree));
zeroOneLossTr = zeros(length(K),length(degree));
for iter = 1 : 30
for deg = 1:length(degree)
for i = 1:length(K)
   setSeed(iter);
   idx = randperm(N);
   Nk = floor(N/K(i));
   idxCV = zeros(K(i), Nk);
   for k = 1:K(i)
      idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
   end
   zeroOneLossTeCV = zeros(K(i), 1);
   zeroOneLossTrCV = zeros(K(i), 1);
   for k = 1:K(i)
       idxTe = idxCV(k,:);
       idxTr = idxCV([1:k-1 k+1:end],:);
       idxTr = idxTr(:);
       yTe = y(idxTe);
       xTe = x(idxTe,:);
       yTr = y(idxTr);
       xTr = x(idxTr,:);
%        %principal component analysis
%        meanXTr = mean(xTr);
%        [COEFF, SCORE, LATENT, TSQUARED] = princomp(xTr);
%        %center test data and try logistic
%        centeredXTe = bsxfun(@minus, xTe, meanXTr);
%        centeredXTeInPCA = centeredXTe * COEFF(:,1:20);
       alpha = 2;
       tXTr = [ones(length(yTr), 1) poly(xTr,degree(deg))];
       tXTe = [ones(length(yTe), 1) poly(xTe,degree(deg))];
       beta = logisticRegression(yTr, tXTr, alpha);
       prediction = 1 - 1 ./ (exp(tXTr * beta) + 1);
       zeroOneLossTrCV(k) = errorZeroOneLoss(yTr, prediction);
       prediction = 1 - 1 ./ (exp(tXTe * beta) + 1);
       %compute number of wrong prediction for test data for KNN model
       zeroOneLossTeCV(k) = BER( prediction,yTe);
   end
   zeroOneLossTr(i,deg) = zeroOneLossTr(i,deg) + mean(zeroOneLossTrCV);
   zeroOneLossTe(i,deg) = zeroOneLossTe(i,deg) + mean(zeroOneLossTeCV);
end
end
end
% zeroOneLossTr = zeroOneLossTr / 30;
% zeroOneLossTe = zeroOneLossTe / 30;

