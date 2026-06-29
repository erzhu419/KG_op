function lem_x = var_x(x, n, sampled, Lem, Lem_s, F_part, key)
%find the variance estimate for solution x
 lem_x = zeros(3,1); 
 idx = x_in_s(sampled, x, n);
 if size(idx,1)==0 %x has not been sampled
  temp = part_id(feat(x, key), F_part); 
  for j=1:3   
   lem_x(j) = Lem{j}(temp(j)); %the variance is determined by feature combination index
  end
 else
   lem_x = Lem_s(:, idx); %the variance is determined separately
 end 
end