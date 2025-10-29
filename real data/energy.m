eta=0.5;
alpha=0.1;
lambda=0.1;
colNames={'Coverage Rate','Operation Time','Average Interval Length'};
rowNames={'Jackknife+','fast Jackknife+','Jackknife-minmax','fast Jackknife-minmax'};
result=zeros(4,3);
filename = 'energy_data.xlsx';
data = readtable(filename);
X = data{:, 1:8};
y = data{:, 9};
n = size(X, 1);
n_t = round(0.8 * n);

X_train = X(1:n_t, :);
Y_train = y(1:n_t);
X_test  = X(n_t + 1:end, :);
Y_test  = y(n_t + 1:end, :);



[n_train,p]=size(X_train);
[n_test,~]=size(X_test);
%Jackknife+
tic;
R_loo=zeros(n_train,1);
B_all=zeros(p,n_train);
pred_jk=zeros(n_test,2);
l_jk=zeros(n_test,1);
for i=1:n_train
    X_i = X_train([1:i-1, i+1:end],:);
    Y_i = Y_train([1:i-1, i+1:end]);
    beta_i=fit_fast(X_i,Y_i,lambda);
    B_all(:,i) = beta_i;
    R_loo(i)=abs(Y_train(i)-X_train(i, :)*beta_i);
end
for j=1:n_test
    upper=zeros(n_train,1);
    lower=zeros(n_train,1);
    for i=1:n_train
        X_i = X_train([1:i-1, i+1:end], :);
        Y_i = Y_train([1:i-1, i+1:end]);
        beta_i=B_all(:,i);
        lower(i)=X_test(j,:)*beta_i-R_loo(i);
        upper(i)=X_test(j,:)*beta_i+R_loo(i);
    end
    pred_jk(j,1)=q_lower(lower,alpha);
    pred_jk(j,2)=q_upper(upper,alpha);
    l_jk(j)=abs(pred_jk(j,1)-pred_jk(j,2));
end
result(1,2)=toc;
result(1,3)=mean(l_jk);
number_jk = 0;
for j=1:n_test
    if (Y_test(j)>=pred_jk(j,1)) && (Y_test(j)<=pred_jk(j,2))
       number_jk=number_jk+1;
    end
end
result(1,1) = number_jk / n_test; 
%fast Jackknife+
tic;
R_hat_loo=zeros(n_train,1);
pred_jkapp=zeros(n_test,2);
l_jkapp=zeros(n_test,1);
B_all_app=zeros(p,n_train);
beta_full_alpha=fit_fast(X_train,Y_train,lambda);
J = X_train'*X_train+hessian(beta_full_alpha,lambda);
for i=1:n_train
    x_i = X_train(i, :);
    r_i = -(Y_train(i) - x_i * beta_full_alpha);
    v = J \ x_i';
    beta_i_app = beta_full_alpha + v * r_i / (1 - x_i * v);
    B_all_app(:,i)=beta_i_app;
    R_hat_loo(i)=abs(Y_train(i)-X_train(i, :)*beta_i_app);
end
for j=1:n_test
    upper_app=zeros(n_train,1);
    lower_app=zeros(n_train,1);
    for i=1:n_train
        beta_i_app = B_all_app(:,i);
        lower_app(i)=X_test(j,:)*beta_i_app-R_hat_loo(i);
        upper_app(i)=X_test(j,:)*beta_i_app+R_hat_loo(i);
    end
    pred_jkapp(j,1)=q_lower(lower_app,alpha);
    pred_jkapp(j,2)=q_upper(upper_app,alpha);
    l_jkapp(j)=abs(pred_jkapp(j,1)-pred_jkapp(j,2));
end
result(2,2)=toc;
result(2,3)=mean(l_jkapp);
number_jkapp=0;
for j=1:n_test
    if (Y_test(j)>=pred_jkapp(j,1)) && (Y_test(j)<=pred_jkapp(j,2))
       number_jkapp=number_jkapp+1;
    end
end
result(2,1)=number_jkapp/n_test;
%original jackknife-minmax
tic;
R_loo=zeros(n_train,1);
B_all  = zeros(p, n_train); 
pred_orjkmm=zeros(n_test,2);
l_jkmm=zeros(n_test,1);
for i=1:n_train
    X_i = X_train([1:i-1, i+1:end], :);
    Y_i = Y_train([1:i-1, i+1:end]);
    beta_i=fit_fast(X_i,Y_i,lambda);
    B_all(:,i) = beta_i;
    R_loo(i)=abs(Y_train(i)-X_train(i, :)*beta_i);
end
for j=1:n_test
    minmax=zeros(n_train,1);
    for i=1:n_train
        X_i = X_train([1:i-1, i+1:end], :);
        Y_i = Y_train([1:i-1, i+1:end]);
        beta_i=B_all(:,i);
        minmax(i)=X_test(j,:)*beta_i;
    end
    pred_orjkmm(j,1)=min(minmax)-q_upper(R_loo,alpha);
    pred_orjkmm(j,2)=max(minmax)+q_upper(R_loo,alpha);
    l_jkmm(j)=abs(pred_orjkmm(j,1)-pred_orjkmm(j,2));
end
result(3,2)=toc;
result(3,3)=mean(l_jkmm);
number_orjkmm=0;
for j=1:n_test
    if (Y_test(j)>=pred_orjkmm(j,1)) && (Y_test(j)<=pred_orjkmm(j,2))
       number_orjkmm=number_orjkmm+1;
    end
end
result(3,1)=number_orjkmm/n_test;
%fast Jackknife-minmax
tic;
B_all=zeros(p,n_train);
R_hat_loo=zeros(n_train,1);
pred_jkmm=zeros(n_test,2);
l_jkapm=zeros(n_test,1);
beta_full_alpha=fit_fast(X_train,Y_train,lambda);
J = X_train'*X_train+hessian(beta_full_alpha,lambda);
for i=1:n_train
    x_i = X_train(i, :);
    r_i = -(Y_train(i) - x_i * beta_full_alpha);
    v = J \ x_i';
    beta_i_app = beta_full_alpha + v * r_i / (1 - x_i * v);
    B_all(:,i) = beta_i_app;
    R_hat_loo(i)=abs(Y_train(i)-X_train(i, :)*beta_i_app);
end
for j=1:n_test
    minmax=zeros(n_train,1);
    for i=1:n_train
        beta_i_app = B_all(:,i) ;
        minmax(i)=X_test(j,:)*beta_i_app;
    end
    pred_jkmm(j,1)=min(minmax)-q_upper(R_hat_loo,alpha);
    pred_jkmm(j,2)=max(minmax)+q_upper(R_hat_loo,alpha);
    l_jkapm(j)=abs(pred_jkmm(j,1)-pred_jkmm(j,2));
end
result(4,2)=toc;
result(4,3)=mean(l_jkapm);
number_jkmm=0;
for j=1:n_test
    if (Y_test(j)>=pred_jkmm(j,1)) && (Y_test(j)<=pred_jkmm(j,2))
       number_jkmm=number_jkmm+1;
    end
end
result(4,1)=number_jkmm/n_test;
T=array2table(result,'RowNames', rowNames, 'VariableNames', colNames);
writetable(T, 'Real Data-energy.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
%Quantile functions
function q_plus = q_upper(v, alpha)
    n = length(v);
    k = ceil((1 - alpha) * (n + 1));
    v_sorted = sort(v);
    k = min(max(k, 1), n); 
    q_plus = v_sorted(k);
end
function q_lower = q_lower(v, alpha)
    n = length(v);
    k = ceil(alpha * (n + 1));
    v_sorted = sort(v);
    k = min(max(k, 1), n);
    q_lower = v_sorted(k);
end
function [theta, fval, exitflag, output] = fit_fast(X, Y, lambda, theta0)
    [n, p] = size(X);
    Y = Y(:);
    if nargin < 4 || isempty(theta0)
        theta0 = zeros(p,1);
    end
    fun = @(th) obj_grad(th, X, Y, lambda);
    opts = optimoptions('fminunc', ...
        'Algorithm','quasi-newton', ...
        'Display','off', ...
        'MaxIterations',1000, ...                 
        'MaxFunctionEvaluations',1e5, ...        
        'OptimalityTolerance',1e-6, ...          
        'FunctionTolerance',1e-6, ...            
        'StepTolerance',1e-6, ...                
        'SpecifyObjectiveGradient',true);

    [theta, fval, exitflag, output] = fminunc(fun, theta0, opts);
end
function [f, g] = obj_grad(theta, X, Y, lambda)
    res = Y - X*theta;
    f_data = 0.5 * (res' * res);
    g_data = -(X' * res);
    t2 = theta.^2;
    s  = sqrt(1 + t2/4);
    reg_val = lambda * ( 2*sum(s - 1) + 0.5*(theta' * theta) );
    f = f_data + reg_val;
    g_reg = lambda * ( theta + theta ./ (2*s) );
    g = g_data + g_reg;
end
function Hreg = hessian(theta, lambda)
    theta = theta(:);
    t2 = theta.^2;
    diag_entries = 1 + 0.5 * (1 + t2/4).^(-3/2);
    Hreg = lambda * diag(diag_entries);
end