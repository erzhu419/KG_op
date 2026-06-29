function  P = perato_con(b, sampled, n, x_L, x_U, key, Lem, Lem_s, F_part, tau_e, alph) 
%return a matrix P whose each column encodes a Pareto optimal solution (satisfy constraints)
 fitnessfcn = @(x)bi_obj(b, sampled, x, key, n);
 options = optimoptions('gamultiobj', 'InitialPopulationMatrix', sampled', 'PopulationSize', 200, 'Display', 'off'); 
 %the above line can be revised! But the PopulationSize should be larger than the size of sampled
 
 P = gamultiobj(fitnessfcn,n, [], [], [],[], zeros(n,1), ones(n,1), @(x)nonlcon(b, sampled, x, key, n, Lem, Lem_s, F_part, tau_e, alph), options);
 P = round(P.*(x_U-x_L)')./(x_U-x_L)'; 
 %for i=1:size(P,1)
   %P(i,:)=min(max(round(P(i,:))', x_L), x_U)';
 %end
end