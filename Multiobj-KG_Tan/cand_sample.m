function S = cand_sample(n_now, n_thr, K1, K2, n, b, B, sampled, key, Lem, Lem_s, F_part, x_L, x_U, tau_e, alph)
%generate the candidate sample set S (each column encodes a selected solution x
 
 %first generate K1 random samples
 S = x_L + diag(x_U-x_L)*lhsdesign(K1,n)';
 S = round(S-x_L)./(x_U-x_L); %normalize interger-valued solutions to [0, 1] in each dimension
 
 %draw K2 times the model coefficients from posterios and generate a set of solutions accordingly 
 for k=1:K2
 bb=b; 
  for i=1:3
   dim=size(key,2)+1;
   BB=B{i}(1:dim,1:dim);
   BB = (BB+BB.')/2; %to avoid numerical error
   ei=eig(BB);
   mi = min(real(ei));
   if (mi < 0) %to avoid numerical error
    BB = BB - 10000*eye(size(BB))*mi;
   end 
    bb{i} = mvnrnd(b{i}(1:dim),BB)';
    bb{i} = [bb{i}; b{i}(dim+1:end)];
  end
    
  if n_now <= n_thr %ignore the constraint if the number of iterations is at most the threshold
    P = perato(bb, sampled, n, x_L, x_U, key);
  else
    P = perato_con(bb, sampled, n, x_L, x_U, key, Lem, Lem_s, F_part, tau_e, alph);
  end
  P = unique(P, 'rows');
  S = union(S', P, 'rows')'; 
 end
 
end
     
     
 
    
    
