function [Mean, Cov, Var, Var_s, Cand, sampled, num, Y] = sim_test(N, n, N0, m, n_thr, Lem0, s0, var0, key, F_part, K1, K2, x_L, x_U, tau_e, alph, stdev)
%% input:
%number of iteration: N
%dimension of the solution: n
%number of pre-samples: N0
%number of previous unique samples used in GA initial population: m
%threshold of #iterations above which the canditate samples need to satisfy the constriant: n_thr
%initial variance estimates for features' combination j for evaluation index i: Lem0{i}(j)
%prior variance of the deviation terms: z0
%weight for the initial variance estimate: s0
%power of feature k for dimension j for evaluation index i of the solution vector: key{i}(k,j) 
%breakpoints for the partition of the feature j for each evaluation index i: F_part{i}{j}
%number of random candidate samples: K1
%number of random draws in postior coefficient for generate candidate samples: K2
%target evaluation index for emissions: tau_e
%standard deviation of the white simulation noise for evaluation i: stdev(i)

%% output:
Mean = cell(N,1); %the posteior mean vectors (b's) of model parameters
Cov = cell(N,1); %the posterior coverance matices (B's) of model parameters 
Var = cell(N,1); %the variance estimate of each partition of the combined feature values
Var_s = cell(N,1); %the variance estimate of those sampled solutions 
Cand = cell(N,1); %the candidate sampling set at each stage 
sampled = []; %the unique sampled solutions (ordered by first-time being sampled) 
num = []; %the number of samples taken for each sampled solution (ordered by first-time being sampled)
Y = zeros(3,N); %the observed evaluation indices of sample xstar at each stage

%% presampling
[b0, B0, z0] = pre_sample(N0, n, x_L, x_U, key, stdev);
%prior mean and covariance for model coeffeicients: b0, B0

%% the main program (N iterations/samples)
 for i = 1:N    
   fprintf('iteration %d\n', i); %print out the current # of stage 
   
   if i==1 %the initial stage
     b = b0;
     B = B0;
     Lem = Lem0;
     Lem_s = [];
   end
   
   %extract the previous m unique samples (if less than m, pick all of them)
   sampled_short=sampled(:, max(size(sampled, 2)-m+1,1):size(sampled, 2));
   
   %generate the candidate solutions for sampling
   cand_sol = cand_sample(i, n_thr, K1, K2, n, b, B, sampled_short, key, Lem, Lem_s, F_part, x_L, x_U, tau_e, alph); 
   K=size(cand_sol,2); %the number of candidate solutions  
   fprintf('the size of the candidate solution set is %d\n', K); %print out K
   
   %compute the variance estimate for each candidate solution
   for k=1:K  
    lem(:, k) = var_x(cand_sol(:,k), n, sampled, Lem, Lem_s, F_part, key);
   end          
   x_star =  KG_sol(n, cand_sol, sampled, b, B, z0, lem, key);  %choose the sample
   %simulation run (read from VISSIM)
   x_star_scale = x_L+x_star.*(x_U-x_L); %transform the solutions beck to original scale 
   y = sim_func(x_star_scale, n, stdev);
   
   %update the mean and covaiance of hyperparameters by the recursive equation
   idx = x_in_s(cand_sol, x_star, n);
   id_xs = x_in_s(sampled, x_star, n);
   [b, B] = update_coeff(b, B, z0, feat(x_star, key), lem(:,idx), id_xs, y);
   
   %update the variance estimate
   [Lem, Lem_s] = update_var(Lem, Lem_s, sampled, num, feat(x_star, key), F_part, key, b, id_xs, y, s0, var0); 
   
   %store the ouput
   Mean{i} = b;
   Cov{i} = B;
   Var{i} = Lem; 
   Var_s{i} = Lem_s;
   Cand{i} = cand_sol;
   Y(:,i) = y; 
      if size(id_xs,1)==0 %x_star has not been sampled
     sampled = [sampled, x_star_scale];
     num(size(sampled,2))=1;
   else
     num(id_xs)=num(id_xs)+1;
   end
 end
 
end