clearvars;
load train/train.mat;
load train/test.mat;

%%Split K fold into train/test, use CNN features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

train.X_cnn = train.X_cnn(:,1:end-1);
test.X_cnn = test.X_cnn(:,1:end-1);

KFold = 5;
[Tr.idxs,Te.idxs] = SplitKFold(KFold ,train.X_cnn);
train.y = (train.y~=4);

Tr.X = train.X_cnn(Tr.idxs,:);
Tr.y = train.y(Tr.idxs);

Te.X = train.X_cnn(Te.idxs,:);
Te.y = train.y(Te.idxs);

%'rbf' Gaussian Kernel 
%'polynomial' Polynomial Kernel 
%'linear' Kernel
Boxcon=[0.1 0.25 0.5 1 2 10];
for i=1:length(Boxcon)
    SVMModel1 = fitcsvm(Tr.X,Tr.y,'KernelFunction','linear','BoxConstraint',Boxcon(i));
    %SVMModel2 = fitcsvm(Tr.X,Tr.y,'KernelFunction','polynomial','BoxConstraint',Boxcon(i));
    %SVMModel3 = fitcsvm(Tr.X,Tr.y,'KernelFunction','rbf','BoxConstraint',Boxcon(i));
    
    label1 = predict(SVMModel1,Te.X);
    %label2 = predict(SVMModel2,Te.X);
    %label3 = predict(SVMModel3,Te.X);

    %Using BER
    predErr1 = BER( label1,Te.y );
    %predErr2 = BER( label2,Te.y );
    %predErr3 = BER( label3,Te.y );
    
    %fprintf('\nTesting error: %.2f%%\n\n', predErr * 100 );
    result(i,1) = predErr1;
    %result(i,2) = predErr2;
    %result(i,3) = predErr3;
end

% result = predict(SVMModel,test.X_cnn);
% 
% Ytest = predict(SVMModel,test.test.X_cnn);
