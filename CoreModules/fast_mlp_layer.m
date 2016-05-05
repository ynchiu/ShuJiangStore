function [ y, dzdw,dzdb ] = fast_mlp_layer( I,weight,bias,dzdy )
%FAST_MLP_LAYER Summary of this function goes here
%   Detailed explanation goes here

%I: input_dim x batch_size

%[in,b]=size(I);    
%[out,in]=size(weight);    
dzdw=[];  
dzdb=[];  
if isempty(dzdy)
    %forward mode
 
    y=weight*I;
    
    if ~isempty(bias)
        
        if numel(bias)==numel(y)
            y=y+bias;%% much faster
        else
            y=bsxfun(@plus,y,bias);
        end
    end
    
else

    if size(I,2)~=size(dzdy,2)
       error('batch size does not agree.'); 
    end
 
    [out,b]=size(dzdy); 
    y=weight'*dzdy;
    if ~isempty(bias)
        %minibatch averaging 
        dzdb=mean(dzdy,2);   
    end
    
    dzdy=permute(dzdy,[1,3,2]);
    %dzdw=zeros(in,out,'like',I);
    I_p=permute(I,[3,1,2]);
    dzdw=bsxfun(@times,dzdy,I_p);
    dzdw=mean(dzdw,3);
    
end




