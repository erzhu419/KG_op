function  P = perato(b, sampled, n, x_L, x_U, key) 
%return a matrix P each column of which encodes a Pareto optimal solution (no constraints)
 fitnessfcn = @(x)bi_obj(b, sampled, x, key, n);
 options = optimoptions('gamultiobj', 'InitialPopulationMatrix', sampled', 'PopulationSize', 200,'Display', 'off'); %can be revised!
 P = gamultiobj(fitnessfcn,n, [], [], [],[], zeros(n,1), ones(n,1), options);
 P = round(P.*(x_U-x_L)')./(x_U-x_L)'; %make sure P is integers 
 %for i=1:size(P,1)
   %P(i,:)=min(max(round(P(i,:))', x_L), x_U)';
 %end
end