%function calculate BER
function error = BER(predict,correct)
  error=0;  
  label = unique(correct);
  N=size(label,1);
  M=size(correct,1);
  unitMatrix=ones(M,1);
  for i=1:N
    correctLab = (correct == label(i));
    predictLab = (predict == label(i));
    numLabel = sum(correctLab);
    mistake = sum(correctLab.* (unitMatrix - predictLab));
    error = error + (mistake/numLabel);
  end
  error=error/N;
end