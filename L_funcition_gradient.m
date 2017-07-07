function [f,df] = L_funcition_gradient(para, X, R, A, theta, rho, T)
%% Data preprocessing
[n, d] = size(X);

L = reshape(para,size(para,1)/d,d);

%% calculate the value of the objective function

LX = L*X';
similar = LX'*LX - theta;
dis = -A.*similar;
big_index=find(dis>50);
temp=log(1+exp(dis));
temp(big_index)=dis(big_index);
temp_adapt=abs(A).*temp;
first_term = sum(temp_adapt(:));
second_term = 0.5*norm(L'-T,'fro')^2;

f = first_term+rho*second_term;


%% calculate the gradient

PP = -A./(1+exp(A.*similar));
first_term =L*X'*(PP+PP')*X;


dL = first_term + rho*(L'-T)';
df = dL(:);

end
