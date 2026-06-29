function id = x_in_s(S, x, n)
%find the "index" (the first apparance) of solution x  in set S (each column encodes a solution)
 K = size(S, 2);

 id = []; j=1;
 %find the first time the k-th solution in S is sampled if any
 while size(id,1)==0&&j<=K 
   if sum(x == S(:, j))==n
      id = j;
   end
     j = j+1;
 end  