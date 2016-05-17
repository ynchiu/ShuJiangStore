function Y = softmaxlogloss(X,c,dzdy)

if(length(size(X))==4)
    sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
    if(sz(1)>1||sz(2)>1) error('Size error in softmax log loss.'); end
    max_c=3;
end
if(length(size(X))==2)
    max_c=1;
end
n_class=size(X,max_c);

Xmax = max(X,[],max_c) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

idx=c(:)'+n_class*[0:size(X,max_c+1)-1];   
if nargin <= 2
    %forward
    t = log(sum(ex,max_c)) +Xmax -reshape(X(idx),size(Xmax));
    Y = sum(t,max_c+1);%%average loss per sample
else
    %bp
    Y = bsxfun(@rdivide, ex, sum(ex,max_c)) ;
    Y(idx)=Y(idx)-1;
    Y = bsxfun(@times, Y, dzdy) ;
end


