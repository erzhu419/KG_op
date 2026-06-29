%% example input and run

%input data example 
N=100; n=5; N0=50; m=40; n_thr=20; s0=1; var0=0.01*ones(3,1); K1=20; K2=2; x_L=zeros(n,1); x_U =100*ones(n,1); tau_e=-0.04; alph=1.645; stdev=0.05*ones(3,1);
for i=1:3
  key{i} = [eye(n), 2*eye(n)];
  length_lem = 1;
  for j=1:size(key{i},2)
    F_part{i}{j} = [0 0.5 1]; %since x is normalized, each feature also normalized between [0, 1]
	length_lem = length_lem*(size(F_part{i}{j},2)-1);
  end
  Lem0{i} = 0.01*ones(length_lem,1);  
end

%run the main program
[Mean, Cov, Var, Var_s, Cand, sampled, num, Y] = sim_test(N, n, N0, m, n_thr, Lem0, s0, var0, key, F_part, K1, K2, x_L, x_U, tau_e, alph, stdev);

%% Post-processing
%plot the true mean function values of the optimized solutions by the algorithm
sampled_short=sampled(:, max(size(sampled, 2)-m+1,1):size(sampled, 2));
P = perato_con(Mean{N}, sampled_short, n, x_L, x_U, key, Var{N}, Var_s{N}, F_part, tau_e, alph);
P = x_L'+P.*(x_U-x_L)'; %transform the solutions beck to original scale 
P = unique(P, 'rows'); 
for i=1:size(P,1)
obj_sol(:,i)=sim_func(P(i,:),n, zeros(3,1));
end
figure; 
scatter(obj_sol(1,:),obj_sol(2,:),'*'); %can add more features to the figure such as axis name and title

%plot the true mean function values of sampled solutions
for i=1:size(sampled, 2)
obj_sample(:,i)=sim_func(sampled(:,i),n,zeros(3,1));
end
hold on
scatter(obj_sample(1,:),obj_sample(2,:),'o');