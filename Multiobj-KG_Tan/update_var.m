function [Lem, Lem_s] = update_var(Lem, Lem_s, sampled, num, F_x, F_part, key, b, first_x, y, s0, var0) 
 %update the variance by partition on combinaitons of feature values
 
 idx=first_x;
 num_x = num(first_x);
 if size(first_x,1)==0 %x has not been sampled
     idx = size(Lem_s, 2)+1;
     Lem_s = [Lem_s, var0];
     num_x = 1;
 end
 K = size(sampled, 2);
 idp = part_id(F_x, F_part); 
 idp_s = zeros(K, 3); %the partition indices for sampled solutions
 
 for k = 1:K 
   idp_s(k,:) = part_id(feat(sampled(:,k), key), F_part); 
 end
 
 for i=1:3
  temp = zeros(1, size(b{i},1)-size(F_x{i},2)); 
  temp(first_x)=1;
  f_t = [F_x{i}, temp];     
  theta = f_t*b{i};
  temp = Lem_s(i, idx);
  Lem_s(i, idx) = (Lem_s(i,idx)*(s0+num_x) + (y(i)-theta)^2)/(s0+num_x+1);
  
  %r is how many times solutions with combination index the same with xstar are sampled
  r = 1; 
  sum_var = 0;
  for k = 1:K 
    if idp_s(k, i) == idp(i)
        sum_var = sum_var + Lem_s(k)*num(k);
        r = r+num(k);
    end
  end  
  sum_var = sum_var + Lem_s(i, idx)*(num_x+1);
  
  Lem{i}(idp(i)) = sum_var/r; %weighted average of the sampled solutions with the same combination index 
 end
end


