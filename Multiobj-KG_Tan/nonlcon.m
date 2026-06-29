function [c, c_eq] = nonlcon(b, sampled, x, key, n, Lem, Lem_s, F_part, tau_e, alph)
%the constraint on the third evaluation index
 temp=x;
 x=zeros(n,1);
 for i=1:n
   x(i)=temp(i);
 end
 F_x = feat(x, key);
 %constraint only defined for each of evaluation index E
 M = size(F_x{3},2); %the number of features used in the surrogate model
 N = size(b{3},1); %the length of vector b; 
 temp = zeros(1, N-M); 
 %find the first time the k-th solution in S is sempled if any
 id = x_in_s(sampled, x, n);
 temp(id)=1; 
 f_t = [F_x{3}, temp]; 
 lem_x = var_x(x, n, sampled, Lem, Lem_s, F_part, key);
 c = f_t*b{3} - tau_e + alph*sqrt(lem_x(3)); %f_t*b{3} <= t_e - alph*stdev.
 c_eq=[];
 
end