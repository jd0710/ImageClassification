function [idxTr,idxTe]=SplitKFold(K,data)
    rng;
    %setSeed(randi(100));
    N = size(data,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    randNum = randi(K);
    idxTe = idxCV(randNum,:);
    idxTe = idxTe(:);
	idxTr = idxCV([1:randNum-1 randNum+1:end],:);
	idxTr = idxTr(:);
end