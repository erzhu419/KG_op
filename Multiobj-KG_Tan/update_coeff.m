function [b_new, B_new] = update_coeff(b, B, z0, F_x, lem_x, first_x, y)
%update model coefficients by the recursive equation 
 b_new = cell(3,1);
 B_new = cell(3,1);
 for i=1:3
  %define two auxiliary quantities
  b_t = b{i};
  B_t = B{i};
  temp = zeros(1, size(b_t,1)-size(F_x{i},2)); 
  temp(first_x)=1;
  f_t = [F_x{i}, temp]; 
    
  if (size(first_x) == 0) %solution x has not been sampled                                                            
    b_t = [b_t; 0]; %add a scalar 0
    f_t = [f_t, 1]; %add a scalar 1
    B_t = [B_t, zeros(size(B_t,1),1)]; %add a new column of 0's 
    B_t = [B_t; zeros(1,size(B_t,2))]; %add a new row of 0's
    B_t(size(B_t,1), size(B_t,2))= z0{i}; %set the value of the last entry 
  end

  ep = y(i) - f_t*b_t;
  ga = lem_x(i) + f_t*B_t*f_t';
    
  b_new{i} = b_t + ep/ga*B_t*f_t'; %update posterior mean of the coefficients 
  B_new{i} = B_t - 1/ga*(B_t*f_t')*f_t*B_t; %update posterior covariance of the coefficients
 end
 
end