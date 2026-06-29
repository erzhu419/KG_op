function [b, B, z0] = pre_sample(N0, n, x_L, x_U, key, stdev)
%using presampling to initialize b (prior mean of coefficients), B (prior covariance of coefficients), and z0 (prior variance of deviation terms)
b = cell(1,3);
B = cell(1,3);
z0 = cell(1,3);
X = cell(1,3);
y = zeros(N0,3);

%generate N0 initial samples 
rng('shuffle');  %this is the place for setting the first random seed!
S = x_L + diag(x_U-x_L)*lhsdesign(N0,n)';
S = round(S-x_L)./(x_U-x_L); %normalize interger-valued solutions to [0, 1] in each dimension

 for k=1:N0
   y(k,:) = sim_func(x_L+S(:, k).*(x_U-x_L), n, stdev);
 end
 for i=1:3
  X{i} = zeros(N0, size(key{i}+1, 2));
  for j = 1:N0
    X{i}(j,1)=1;  
    for jj=1:size(key{i}, 2)
      X{i}(j,jj+1) = prod(S(:,j).^key{i}(:,jj));
    end
  end
  b{i} = regress(y(:,i), X{i});
 end

 for i=1:3
   B{i}= var(b{i})*eye(size(key{i},2)+1);
   z0{i} = var(y(:,i)-X{i}*b{i});
 end

end

