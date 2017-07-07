clear;clc;
% Load the data set, 
tic;
tmp = load('ionosphere.data');
y = tmp(:,end);
X = tmp(:,1:(end-1));
% X = [tmp(:,1) tmp(:,3:(end-1))];
clear tmp;

n_fold=4;

tt=length(y);
rp = randperm(tt);
y = y(rp);
X = X(rp, :);

% Uncomment below two lines to verify the power of our method on
% feature selection. Note the parameters of ml_admm may be need to tune. 
% Sorry! ::>_<:: I lost my best parameter setting;

% noise=zeros(tt,100); 
% X = [X noise];

%% ours
acc = zeros(1,n_fold);
for i=1:n_fold
    train_start = ceil(tt/n_fold * (i-1)) + 1;
    train_end = ceil(tt/n_fold * i);
    
    yt = [];
    Xt = zeros(0, size(X,2));
    if (i > 1);
        yt = y(1:train_start-1);
        Xt = X(1:train_start-1,:);
    end
    if (i < n_fold),
        yt = [yt; y(train_end+1:length(y))];
        Xt = [Xt; X(train_end+1:length(y), :)];
    end
    
    nt = length(yt);
    yt = yt(1:nt);
    Xt = Xt(1:nt, :);
    XT = X(train_start:train_end, :);
    yT = y(train_start:train_end);

    X1=repmat(1:length(yT),length(yT),1);
    X2=X1';
    R=[X1(:),X2(:),sign((yT(X1(:))==yT(X2(:)))-0.5)];
    
    W = ml_admm(XT, R, 1, 0.05, 0.1, 30, 1, 1);
    feature_number=length(find(sum(W,1)~=0));
    error=0;
    for j = 1:size(Xt,1)
        classifyresult = KNN(Xt(j,:),XT, W, yT, 4);
        fprintf('The prediction is: %d The Ground truth is: %d\n',[classifyresult yt(j)])
        if(classifyresult~=yt(j)),
            error = error+1;
        end
    end
    acc(i)=1-error/size(Xt,1);
end
fprintf('Accuracy: %f\n',mean(acc))

toc;