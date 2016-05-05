function [  net,res,opts ] = sgd(  net,res,opts )
%NET_APPLY_GRAD_SGD Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end
    
    if ~isfield(opts.parameters,'clip')
        opts.parameters.clip=0;
    end
    
    if (~isfield(net,'mom_factor'))
        net.mom_factor=0; 
    end
    
    if ~isfield(opts,'results')||~isfield(opts.results,'lrs')
        opts.results.lrs=[];%%not really necessary
    end
    opts.results.lrs=[opts.results.lrs;gather(opts.parameters.lr)];
    
    
    net.mom_factor=net.mom_factor*opts.parameters.mom+(1-opts.parameters.mom);
    
    for layer=1:numel(net.layers)
        if strcmp(net.layers{layer}.type,'conv')||strcmp(net.layers{layer}.type,'mlp')
            
            if opts.parameters.clip>0
                mask=abs(res(layer).dzdw)>opts.parameters.clip;
                res(layer).dzdw(mask)=sign(res(layer).dzdw(mask)).*opts.parameters.clip;%%this type of processing seems to be very helpful
                mask=abs(res(layer).dzdb)>opts.parameters.clip;
                res(layer).dzdb(mask)=sign(res(layer).dzdb(mask)).*opts.parameters.clip;
            end
            net.layers{1,layer}.momentum{1}=opts.parameters.mom.*net.layers{1,layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw- opts.parameters.weightDecay * net.layers{1,layer}.weights{1};
            net.layers{1,layer}.weights{1}=net.layers{1,layer}.weights{1}+opts.parameters.lr*net.layers{1,layer}.momentum{1}./net.mom_factor;
            
            net.layers{1,layer}.momentum{2}=opts.parameters.mom.*net.layers{1,layer}.momentum{2}-(1-opts.parameters.mom).*res(layer).dzdb;
            net.layers{1,layer}.weights{2}=net.layers{1,layer}.weights{2}+opts.parameters.lr*net.layers{1,layer}.momentum{2}./net.mom_factor;

        end
    end
   
end

