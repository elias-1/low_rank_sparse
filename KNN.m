function relustLabel = KNN(inx,data,W,labels,k)
%%  
% inx is data to test, data is training data, labels is training labels
%%

[datarow , datacol] = size(data);
inX = repmat(inx,[datarow,1]);
% M=W'*W; 
% similar = diag(inX*M*data');
X_transfm = W*inX';
Y_transfm = W*data';
X_Y=X_transfm .* Y_transfm;
similar=sum(X_Y,1);
% similar1-similar
[B , IX] = sort(similar,'descend');
len = min(k,length(B));
relustLabel = mode(labels(IX(1:len)));
end
