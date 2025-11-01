rng(98765);
n_train=100;
n_test=100;
Round=50;
lambda=0.1;
P_number=[50,100,200];
alpha=0.1;
colNames={'Coverage Rate','Operation Time','Intervsl Length','Jaccard Index'};
rowNames={'(n=100,p=50,λ=0.1)','(n=100,p=100,λ=0.1)','(n=100,p=200,λ=0.1)'};
result=zeros(3,4);
result_app=zeros(3,4);
group=0;
for p=P_number
    group=group+1;
    total_time=zeros(Round,1);
    total_time_app=zeros(Round,1);
    Accuracy=zeros(Round,1);
    Accuracy_app=zeros(Round,1);
    length_jk=zeros(Round,1);
    length_jkapp=zeros(Round,1);
    j_index=zeros(Round,1);
    for r=1:Round
        number_jk=0;
        number_jkapp=0;
        j_i=zeros(n_test,1);
        beta = randn(p, 1);
        X=randn(n_train,p)/ sqrt(p);
        Y_train = X * beta + randn(n_train, 1);
        X_test=randn(n_test,p)/ sqrt(p);
        Y_test=X_test*beta+randn(n_test,1);
        R_loo=zeros(n_train,1);%originial R_loo
        R_hat_loo=zeros(n_train,1);%R_loo constructed by approximate estimator
        pred_jk=zeros(n_test,2);%prediction intervals by original jackknife+
        pred_jkapp=zeros(n_test,2);%prediction intervals by fast jackknife+
        l_jk=zeros(n_test,1);
        l_jkapp=zeros(n_test,1);
        %Original Jackknife+
        tic;
        B_all  = zeros(p, n_train); 
        for i=1:n_train
            X_i = X([1:i-1, i+1:end], :);
            Y_i = Y_train([1:i-1, i+1:end]);
            beta_i=fit_fast(X_i,Y_i,lambda);
            B_all(:,i) = beta_i;
            R_loo(i)=abs(Y_train(i)-X(i, :)*beta_i);
        end
        for j=1:n_test
            upper=zeros(n_train,1);
            lower=zeros(n_train,1);
            for i=1:n_train
                X_i = X([1:i-1, i+1:end], :);
                Y_i = Y_train([1:i-1, i+1:end]);
                beta_i=B_all(:,i);
                lower(i)=X_test(j,:)*beta_i-R_loo(i);
                upper(i)=X_test(j,:)*beta_i+R_loo(i);
            end
            pred_jk(j,1)=q_lower(lower,alpha);
            pred_jk(j,2)=q_upper(upper,alpha);
            l_jk(j)=abs(pred_jk(j,1)-pred_jk(j,2));
        end
        total_time(r) = toc;
        %jackknife+ with approximate leave-one-out estimators
        tic;
        B_all_app=zeros(p,n_train);
        beta_full_alpha=fit_fast(X,Y_train,lambda);
        J = X'*X+hessian(beta_full_alpha,lambda);
        for i=1:n_train
            x_i = X(i, :);
            r_i = -(Y_train(i) - x_i * beta_full_alpha);
            v = J \ x_i';
            beta_i_app = beta_full_alpha + v * r_i / (1 - x_i * v);
            B_all_app(:,i)=beta_i_app;
            R_hat_loo(i)=abs(Y_train(i)-X(i, :)*beta_i_app);
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
        total_time_app(r)=toc;
        R_loo=zeros(n_train,1);
        %accuracy
        for j=1:n_test
            if (Y_test(j)>=pred_jk(j,1)) && (Y_test(j)<=pred_jk(j,2))
                number_jk=number_jk+1;
            end
            if (Y_test(j)>=pred_jkapp(j,1)) && (Y_test(j)<=pred_jkapp(j,2))
                number_jkapp=number_jkapp+1;
            end
        end
        for j=1:n_test
            S_1=max(0,min(pred_jk(j,2),pred_jkapp(j,2))-max(pred_jk(j,1),pred_jkapp(j,1)));
            S_2=max(pred_jk(j,2),pred_jkapp(j,2))-min(pred_jk(j,1),pred_jkapp(j,1));
            j_i(j)=S_1/S_2;
        end
        Accuracy(r)=number_jk/100;
        Accuracy_app(r)=number_jkapp/100;
        length_jk(r)=mean(l_jk);
        length_jkapp(r)=mean(l_jkapp);
        j_index(r)=mean(j_i);
    end
    result(group,1)=mean(Accuracy);
    result(group,2)=mean(total_time);
    result(group,3)=mean(length_jk);
    result(group,4)=mean(j_index);
    result_app(group,1)=mean(Accuracy_app);
    result_app(group,2)=mean(total_time_app);
    result_app(group,3)=mean(length_jkapp);
    result_app(group,4)=mean(j_index);
end
disp('Result for Original Jackknife+')
T_1=array2table(result, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_1);
disp('Result for Jackknife+ by Approximate estimators')
T_2=array2table(result_app, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_2);
writetable(T_1, 'JKplus_Results_lambda01.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
writetable(T_2, 'JKplus_Results_app_lambda01.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
%quantile functions
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
