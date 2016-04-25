function beta = penLogisticRegression(y, tX, alpha, lambda)
    %initialize
    [~,dimension] = size(tX);
    beta = ones(dimension,1);
    %algorithm parameters
    maxIters = 1000;
    converged = 1e-10;
    %iterate
    cost = zeros(maxIters,1);
    for k = 1:maxIters
       %compute gradient
       tXTimesBeta = tX*beta;
       sigma = exp(tXTimesBeta)./(1+exp(tXTimesBeta));
       for j = 1:length(y)
          if(isnan(sigma(j)))
              sigma(j) = 1;
          end
       end
       gradient = tX' * (sigma - y) + 2 * lambda * [0;beta(2:end,1)];
       %compute cost
       cost(k+1) = - y' * tXTimesBeta + sum(log(1+exp(tXTimesBeta))) + lambda * sum(beta.^2);
       %gradient descent update beta
       beta = beta - alpha * gradient;
       %convergence judgement
       if(k > 1 && abs(cost(k+1) - cost(k)) < converged)
            break;
       end
    end
end