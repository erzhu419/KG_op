function log_KG = KG_factor(n, S, sampled, x, b, B, key, z0, lem_x)  
%compute the logrithm of the KG factor for solution x

 K = size(S,2); %the number of candidate solutions
 %F is the feature values for all solution in S, each cell entry of F is for one evaluation index  
 F = cell(3,1);
 for k =1:K
  F_x = feat(S(:,k), key);
  for i=1:2
   F{i}(k, :) = F_x{i}; 
  end
 end
 
 for i=1:2 %for each of the evaluation indices (D:i=1, A:i=2)
  M = size(F{i},2); %the number of features used in the surrogate model
  N = size(b{i},1); %the length of vector b
  F_t = zeros(K, size(b{i},1)); 
  for k = 1:K
    temp = zeros(1, N-M); 
    %find the first time the k-th solution in S is sempled if any
    id = x_in_s(sampled, S(:,k), n);
    temp(id)=1;  
    f_t = [F{i}(k, :), temp]; 
    F_t(k,:) = f_t;
  end
  b_t= b{i}; 
  B_t = B{i};
 
 %find the index of solution x in set S (i.e., column index)
  id = x_in_s(sampled, x, n);
  idx = x_in_s(S, x, n);
  if size(id,1)==0 %if x has not been sampled, then update f_t, F_t, b_t and B_t
     new_col = zeros(K,1); % add a new column of 0's to D
     new_col(idx)=1;         % set the value of the entry corresponding to the deviation term to 1
     F_t = [F_t, new_col];
     b_t = [b_t; 0];
     B_t =[B_t,  zeros(size(B_t,1),1)
           zeros(1, size(B_t,2)), z0{i}];  
     f_t = F_t(idx,:); 
  end
    
  p = -F_t*b_t; %add the minus sign to change the min. problem to max. problem
  q = F_t*B_t*f_t' / sqrt(lem_x(i) + f_t*B_t*f_t');
  pq = sortrows([p,q],2);%sort the sequence of pairs(pj, qj) so that the qj are in nondecreasing order and ties in q are broken so that pj<=pj+1 if qj=qj+1 
    
  %if qj = qj+1, first ensure pj <= pj+1, and then remove entry j, forming new (pj, qj) pairs 
  id = zeros(K,1);
  for j=1:K-1
   %if abs(pq(j,2)-pq(j+1,2))<=0.0000001
   if pq(j,2)== pq(j+1,2)
      id(j) = j;
     if (pq(j,1)>pq(j+1,1)) %make sure pj<=pj+1 if qj=qj+1
        temp = pq(j,1);
        pq(j,1) = pq(j+1,1);
        pq(j+1,1) =temp;
     end 
   end 
  end
  
  pq((id>0), :)=[]; %remove the (pj, qj) pair if qj=qj+1
  % form new p, q pairs
  p = pq(:,1);
  q = pq(:,2);
   
  %calculate the nondecreasing sequence vector a associated with pq
  dim = size(pq,1);
  a = zeros(dim,1);
  a(dim)=inf; %a0 =-inf; aJ=inf
  for j=1:1:dim-1
    a(j,1) = (p(j)-p(j+1))/(q(j+1)-q(j));
  end
  pqa = [p, q, a];
   
  %remove entry j if aj>=aj+1, forming new pqa
  id = zeros(dim,1);
  for j=1:dim-1
   %if abs(pq(i,2)-pq(i+1,2))<=0.0000001
   if pqa(j,3) >= pqa(j+1,3)
      id(j) = j;
   end 
  end
  
  pqa((id>0), :)=[]; %remove the (pj, qj) pair if qj=qj+1
  %form new p, q pairs
  q = pqa(:,2);
  a = pqa(:,3);
  
  sum_v=0;
  for j=1:size(pqa,1)-1
     sum_v = sum_v +(q(j+1)-q(j))*(normpdf(-abs(a(j)))-abs(a(j))*normcdf(-abs(a(j))));
  end
  log_KG(i) = log(sum_v);
 end
 
end