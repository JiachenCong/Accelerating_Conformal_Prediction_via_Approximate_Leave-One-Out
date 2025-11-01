rng(98765);
n_train=100;
n_test=100;
Round=50;
lambda=0.1;
P_number=[50,100,200];
alpha=0.1;
colNames={'Coverage Rate','Operation Time','Intervsl Length','Jaccard Index'};
rowNames={'(n=100,p=50,λ=0.1)','(n=100,p=100,λ=0.1)','(n=100,p=200,λ=0.1)'};
result_mm=zeros(3,4);
result_orjkmm=zeros(3,4);
group=0;
for p=P_number
    group=group+1;
    total_time_mm=zeros(Round,1);
    total_time_orjkmm=zeros(Round,1);
    Accuracy_mm=zeros(Round,1);
    Accuracy_orjkmm=zeros(Round,1);
    length_jkmm=zeros(Round,1);
    length_jkapm=zeros(Round,1);
    j_index=zeros(Round,1);
    for r=1:Round
        number_jkmm=0;
        number_orjkmm=0;
        j_i=zeros(n_test,1);
        beta = randn(p, 1);
        X=randn(n_train,p)/ sqrt(p);
        Y_train = X * beta + randn(n_train, 1);
        X_test=randn(n_test,p)/ sqrt(p);
        Y_test=X_test*beta+randn(n_test,1);
        R_loo=zeros(n_train,1);%originial R_loo
        R_hat_loo=zeros(n_train,1);%R_loo constructed by approximate estimator
        pred_jkmm=zeros(n_test,2);%prediction intervals by fast jackknife-minmax
        pred_orjkmm=zeros(n_test,2);%prediction intervals by original jackknife-minmax
        l_jkmm=zeros(n_test,1);
        l_jkapm=zeros(n_test,1);
        %original jackknife-minmax
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
            minmax=zeros(n_train,1);
            for i=1:n_train
                X_i = X([1:i-1, i+1:end], :);
                Y_i = Y_train([1:i-1, i+1:end]);
                beta_i=B_all(:,i);
                minmax(i)=X_test(j,:)*beta_i;
            end
            pred_orjkmm(j,1)=min(minmax)-q_upper(R_loo,alpha);
            pred_orjkmm(j,2)=max(minmax)+q_upper(R_loo,alpha);
            l_jkmm(j)=abs(pred_orjkmm(j,1)-pred_orjkmm(j,2));
        end
        total_time_orjkmm(r) = toc;
        %jackknife-minmax with approximate leave-one-out estimators
        tic;
        B_all=zeros(p,n_train);
        beta_full_alpha=fit_fast(X,Y_train,lambda);
        J = X'*X+hessian(beta_full_alpha,lambda);
        for i=1:n_train
            x_i = X(i, :);
            r_i = -(Y_train(i) - x_i * beta_full_alpha);
            v = J \ x_i';
            beta_i_app = beta_full_alpha + v * r_i / (1 - x_i * v);
            B_all(:,i) = beta_i_app;
            R_hat_loo(i)=abs(Y_train(i)-X(i, :)*beta_i_app);
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
        total_time_mm(r)=toc;
        %accuracy
        for j=1:n_test
            if (Y_test(j)>=pred_jkmm(j,1)) && (Y_test(j)<=pred_jkmm(j,2))
                number_jkmm=number_jkmm+1;
            end
            if (Y_test(j)>=pred_orjkmm(j,1)) && (Y_test(j)<=pred_orjkmm(j,2))
                number_orjkmm=number_orjkmm+1;
            end
        end
        for j=1:n_test
            S_1=max(0,min(pred_orjkmm(j,2),pred_jkmm(j,2))-max(pred_orjkmm(j,1),pred_jkmm(j,1)));
            S_2=max(pred_orjkmm(j,2),pred_jkmm(j,2))-min(pred_orjkmm(j,1),pred_jkmm(j,1));
            j_i(j)=S_1/S_2;
        end
        Accuracy_mm(r)=number_jkmm/100;
        Accuracy_orjkmm(r)=number_orjkmm/100;
        length_jkmm(r)=mean(l_jkmm);
        length_jkapm(r)=mean(l_jkapm);
        j_index(r)=mean(j_i);
    end
    result_mm(group,1)=mean(Accuracy_mm);
    result_mm(group,2)=mean(total_time_mm);
    result_mm(group,3)=mean(length_jkapm);
    result_mm(group,4)=mean(j_index);
    result_orjkmm(group,1)=mean(Accuracy_orjkmm);
    result_orjkmm(group,2)=mean(total_time_orjkmm);
    result_orjkmm(group,3)=mean(length_jkmm);
    result_orjkmm(group,4)=mean(j_index);
end
disp('Result for Jackknife-minmax by Approximate estimators')
T_3=array2table(result_mm, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_3);
disp('Result for Original Jackknife-minmax')
T_4=array2table(result_orjkmm, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T_4);
writetable(T_3, 'JKplus_lambda01_appmm.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);
writetable(T_4, 'JKplus_lambda01_orjkmm.xlsx', 'Sheet', 'Sheet1', 'WriteRowNames', true);

function q_plus = q_upper(v, alpha)
    n = length(v);
    k = ceil((1 - alpha) * (n + 1));
    v_sorted = sort(v);
    k = min(max(k, 1), n); 
    q_plus = v_sorted(k);
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
