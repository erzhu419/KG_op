function x_star = KG_sol(n, S, sampled, b, B, z0, lem, key)    
%compute the sampling decision

%% first compute the set of solutions with non-dominant KG values
 P_KG = []; %will be used to store solutions x with non-dominated KG factors  
 K = size(S,2); %the number of candidate solutions
 V_KG = [[1:K]', zeros(K, 2)]; %column 2 and 3 corresponds to evaluation index 'D' and 'A'
 %record the calculated logKG factor for each solution considered 
 for k = 1:K
    x = S(:,k); 
    log_KG = KG_factor(n, S, sampled, x, b, B, key, z0, lem(:, k));
    V_KG(k, 2:3)=log_KG;
 end
 V_KG = sortrows(V_KG, 2); %sort rows of V_KG by column 2
 L_KG = []; %used to store log_KG values of nondominate solutions
 for k=1:K
    if k==K || max(V_KG(k+1:K, 3)) <= V_KG(k, 3)
        x = S(:,V_KG(k,1)); 
        P_KG = [P_KG, x];
        L_KG = [L_KG; V_KG(k,2:3)];
    end
 end

%% pick a final solution from the set P_KG
%option1: randomly choose one
%  x_star = P_KG(:, randi(size(P_KG, 2))); 

%option2: use weighted average rule (can change weights manually, the example here is equal weights)
  [L_KG_new, ind_x] = sort(-L_KG*[0.5; 0.5]);
  x_star = P_KG(:, ind_x(1));  %pick the solution with largest weight average bi-objective values
 
%option 3: can use more complicated rule for picking a solution x (below is based on maximum crowding distance)
%  [CD, ind_x] = sort(-crowding_distance([1:size(P_KG,2)],L_KG));
%  if size(CD,2)>2
%    ind_x_star = ind_x(3); %if there is more than two Perato optimal solutions, pick the one with maximum finite CD
%  else
%    ind_x_star = ind_x(randi(size(CD,2))); 
%  end
%  x_star = P_KG(:, ind_x_star); 
end
 




