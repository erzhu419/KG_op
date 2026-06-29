function idp = part_id(F_x, F_part)
%get the feature combination index of solution x
 idp = zeros(1,3);
 part=cell(3,1); 
 for i=1:3
  %number of features of the i-th evaluation index
  M = size(F_x{i},2)-1;
  N = zeros(1,M); %record the number of breakpoints for each of the M features
  for j=1:M
   temp = F_part{i}{j}; %the breakpoints of the j-th feature for evaluation index i
   %number of breakpoints for the value range of the j-th component of the i-th evaluation index
   N(j) = size(temp, 2); 
   for k=1:N(j)-1
    if k == 1
      if F_x{i}(j) >= temp(k) &&  F_x{i}(j) <= temp(k+1)
        part{i}(j) = k; 
      end
    else
      if F_x{i}(j) > temp(k)  &&  F_x{i}(j) <= temp(k+1)
        part{i}(j) = k;   
      end
    end
   end
  end
  for j=1:M-1
    idp(i) = idp(i)+(part{i}(j)-1)*prod(N(j+1:M)-1);
  end
  idp(i) = idp(i)+part{i}(M);
 end
 
end
    
  
  
      