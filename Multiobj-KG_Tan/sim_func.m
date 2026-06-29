function y = sim_func(x,n,stdev)
%simulation for decision vector x with dimension n (here noises is additive)
%% ZDT2 probem as an example(with 0 to 100 integer decision variables) 
  y(1) = x(1)/100+normrnd(0, stdev(1)); %1st objective function
  g = 1;
  for j=2:n
    g=g+9/(n-1)*x(j)/100;
  end
  y(2) = g*(1-(x(1)/g/100)^2)+normrnd(0, stdev(2)); %2nd objective function
  y(3) = -(x(1)/100-0.5)^2+normrnd(0, stdev(3)); %constraint function
end