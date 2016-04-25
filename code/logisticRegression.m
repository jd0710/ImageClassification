%logistic regression using gradient descent
function beta = logisticRegression(y, tX, alpha)
    %initialize
    [~,dimension] = size(tX);
    beta = ones(dimension,1);
    %algorithm parameters
    maxIters = 1000;
    converged = 1e-15;
    %iterate
    for k = 1:maxIters
       %compute gradient
       tXTimesBeta = tX*beta;
       sigma = exp(tXTimesBeta)./(1+exp(tXTimesBeta));
       for j = 1:length(y)
          if(isnan(sigma(j)))
              sigma(j) = 1;
          end
       end
       gradient = tX' * (sigma - y);
       %compute cost
       cost = - y' * tXTimesBeta + sum(log(1+exp(tXTimesBeta)));
       %gradient descent update beta
       beta = beta - alpha * gradient;
       %convergence judgement
       if(k > 1 && abs(cost - lastCost) < converged)
            break;
       end
       %store last cost
       lastCost = cost;
    end
end