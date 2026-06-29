function obj = bi_obj(b, sampled, x, key, n)
%compute the posteior mean of bi-objective values 
 temp=x;
 x=zeros(n,1);
 for i=1:n
   x(i)=temp(i);
 end
 f_t = cell(1,2);
 F_x = feat(x, key); 
 for i=1:2 %for each of the evaluation indices (D:i=1, A:i=2)
  M = size(F_x{i},2); %the number of features used in the surrogate model
  N = size(b{i},1); %the length of vector b; 
  temp = zeros(1, N-M); 
  %find the first time solution x is sempled if any
  id = x_in_s(sampled, x, n);
  temp(id)=1;  
  f_t{i}= [F_x{i}, temp];
 end
 for i=1:2
  %obj(i)=f_t{i}*b{i};
  %can use below instead if we are indifferent about measurement values to some decimal point 
  obj(i)=round(f_t{i}*b{i}*100)/100; %here 100 is an example, can be changed to other value when needed
 end
 
end