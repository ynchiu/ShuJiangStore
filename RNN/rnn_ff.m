function [ net,res,opts ] = rnn_ff( net,inputs,opts )
%NET_FF Summary of this function goes here
%   Detailed explanation goes here
%this implementation can be simplified by merging hidden and input
    
    if opts.use_gpu
        inputs.data=gpuArray(single(inputs.data));
        if isfield(inputs,'labels')
            inputs.labels=gpuArray(single(inputs.labels));
        end
        if isfield(inputs,'predicts')
            inputs.predicts=gpuArray(single(inputs.predicts));
        end
    end
    
    n_frames=opts.parameters.n_frames;    
    n_hidden_nodes=opts.parameters.n_hidden_nodes;
    batch_size=opts.parameters.batch_size;
    
    res.Hidden{1}=zeros(n_hidden_nodes,batch_size,'like',inputs.data);
 
    if isfield(inputs,'labels')
        opts.err=zeros(2,n_frames,'like',inputs.data);
        opts.loss=zeros(1,n_frames,'like',inputs.data);
    end

    
    for f=1:n_frames
        %Process inputs
        res.Input{f}(1).x=[res.Hidden{f};inputs.data(:,:,f)];%inputs
        
        %Input transform
        [ net{1},res.Input{f},opts ] = net_ff( net{1},res.Input{f},opts ); 
        
        %Update hidden nodes;
        res.Hidden{f+1}=opts.parameters.Id_w.*res.Hidden{f} + res.Input{f}(end).x;
        
        %Data fitting transform
        res.Fit{f}(1).x=res.Hidden{f+1};
        if isfield(inputs,'predicts')
            res.Fit{f}(1).predicts=inputs.predicts(:,:,f);
        end
        if isfield(inputs,'labels')
            res.Fit{f}(1).class=inputs.labels(:,f);
        end
        
        [ net{2},res.Fit{f},opts ] = net_ff( net{2},res.Fit{f},opts ); 
        if isfield(inputs,'labels')
            opts.err(:,f)=error_multiclass(res.Fit{f}(1).class,res.Fit{f});          
        end
        
        opts.loss(:,f)=mean(res.Fit{f}(end).x(:));
        
    end
    if isfield(inputs,'labels')
        opts.err=mean(opts.err,2)./opts.parameters.batch_size;
    end
    opts.loss=mean(opts.loss(:));
end

