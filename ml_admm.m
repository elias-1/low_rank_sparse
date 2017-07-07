function [W, history] = ml_admm(X, R, theta, lambda, rho, maxit, verbose, alpha)

%% USAGE: [W,history] = ml_admm(X, R, theta, lambda, rho, maxit, verbose, alpha)
% metric learning using admm method.
%
% solves the following problem via ADMM:
%   minimize sum_(i,j)->R (log(1+exp(-Aij(X(i,:)'L'LX(j,:)-theta))) +
%   lambda*\|L\|_{2,1}
%
%% Input:
%        X --- N x d matrix, n data points in a d-dim space
%        R --- D x 3 supervision information,denotes (x,y,Aij),D is the total number of the
%        pairwise training data 
%        theta ---  a threshold controls similarity
%        lambda --- a positive coefficient for regulerization on L2-1-norm of L
%        rho --- a positve coefficient introduced by ADMM
%        maxit --- maximum number of iterations of function minimize until stop (default=30)
%        verbose --- whether to verbosely display the learning process
%                    (default=false)
%        alpha --- over-relaxation parameter (typical values for alpha are
%        between 1.0 and 1.8).
%
%% Output:
%        L --- d x d
%        history --- a structure that contains the objective value, the primal and
%                    dual residual norms, and the tolerances for the primal and 
%                    dual residual norms at each iteration.
%
%
%% Reference:
%    http://web.stanford.edu/~boyd/papers/admm/group_lasso/group_lasso.html
%   Copyright (C) 2016 by Shilei Cao.


t_start = tic;


%% Global constants and defaults
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

%% Data preprocessing

[n,d] = size(X);

loss_func = 'L_funcition_gradient';

% A is used to calculate the gradient of the function,not to the function.
A=zeros(n,n);
for i=1:size(R,1),
    A(R(i,1),R(i,2))=R(i,3);
    A(R(i,2),R(i,1))=R(i,3);
end

A(sub2ind([n,n],1:n,1:n))=0;


%% ADMM solver
r = 13;
% r=d;
L = randn(r,d);
% L = zeros(r,d);
W = zeros(d,r);
U = zeros(d,r);

if verbose
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end


for k = 1:MAX_ITER
   
%      L - update
   Lstarbest = minimize(L(:), loss_func, maxit, 1, X, R, A, theta , rho, W-U);
   L = reshape(Lstarbest,r,d);
   if lambda==0
       continue
   end

%     W - update
    Wold =W;
    L_hat = alpha*L' + (1-alpha)*Wold;
    W = shrinkage(L_hat+U,lambda/rho);
 
%     U - update
    U = U + (L_hat-W);
    
%      diagnostics, reporting, termination checks
    history.objval(k) = objective(X,lambda, L, W, R, A, theta);
    LL=L';
    history.r_norm(k)  = norm(LL(:) - W(:));
    history.s_norm(k)  = norm(-rho*(W(:) - Wold(:)));

    history.eps_pri(k) = d*ABSTOL + RELTOL*max(norm(LL(:)), norm(-W(:)));
    history.eps_dual(k)= d*ABSTOL + RELTOL*norm(rho*U(:));

    if verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
%     if history.r_norm(k) > 2*history.s_norm(k),
%         rho=2*rho;
%     elseif 2*history.r_norm(k) < history.s_norm(k)
%         rho=rho/2;
%     end
    feature_number=length(find(sum(W,2)~=0))
    if feature_number>30 && feature_number<40
        break;
    end
    
end

W=W';

if lambda==0
    W=L;
end
if verbose
    toc(t_start);
end

end

function [p]= objective(X,lambda, L, W, R, A, theta)
    LX = L*X';
    similar = LX'*LX - theta;
    dis = -A.*similar;
    big_index=find(dis>50);
    temp=log(1+exp(dis));
    temp(big_index)=dis(big_index);
    temp_adapt=abs(A).*temp;
    first_term = sum(temp_adapt(:))
    second_term =sum(sqrt(sum(abs(W).^2,2)))
    
    p = first_term + lambda*second_term;
    
end


function W = shrinkage(X, kappa)
   W = repmat(max(zeros(size(X,1),1),(ones(size(X,1),1) - kappa./sqrt(sum(abs(X).^2,2)))),1,size(X,2)).*X;
end