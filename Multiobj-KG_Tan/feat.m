function F_x = feat(x, key)
%compute the feature list of solution x, F_x, each evaluation index in each cell
%x is a column vector
%index of key dimension j for evaluation index i of the solution vector: key(i,j) 
 F_x = cell(3,1);
 for i=1:3  %evaluation index i=1,2,3 represent 'D', 'A', 'E', respectively
   M = size(key{i},2); %the number of feasures for evaluation index i
   F_x{i}(1)=1;
   for j=1:M
     %this is a simple example of feature function using polynomials  
      F_x{i}(j+1) = prod(x.^key{i}(:,j));
   end
 end
  
end