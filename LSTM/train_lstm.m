function [net,opts]=train_lstm(net,opts)

    opts.training=1;

    
    if ~isfield(opts.parameters,'learning_method')
        opts.parameters.learning_method='sgd';
    end
    
    if ~isfield(opts,'display_msg')
        opts.display_msg=1; 
    end
    opts.MiniBatchError=[];
    opts.MiniBatchLoss=[];

    
    tic
    
    opts.order=randperm(opts.n_train);
    
    
    
    for mini_b=1:opts.n_batch
                
        idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);

        %%get your input here
        
        if ~isfield(opts,'train_labels')
            inputs.data=opts.train(:,idx,:);
            output_data=inputs.data;
            inputs.data=inputs.data+randn(size(inputs.data))*0.005;
        else
            %%%fill in the details
            inputs.data=opts.train(:,idx,:);
            inputs.labels=opts.train_labels(idx,:);
        end
        
        
        %forward
        [ net,res,opts ] = lstm_ff( net,inputs,opts );

        %%%get your gradients;
        if ~isfield(opts,'train_labels')
            
            loss=0;
            for f=1:opts.parameters.n_frames
               opts.lstm_dzdy{f}=res.Fit{f}(end).x -output_data(:,:,f); 
               loss=loss+sum(opts.lstm_dzdy{f}(:).^2)./opts.parameters.batch_size;
            end
            
        else
            %this is what we do with softmax log-loss
            for f=1:opts.parameters.n_frames
            
                opts.lstm_dzdy{f}=single(1.0);        
                if opts.use_gpu
                    opts.lstm_dzdy{f}=gpuArray(single(opts.lstm_dzdy{f}));
                end
            end
            
            
        end
        
        %%backward
        [ net,res,opts ] = lstm_bp( net,res,opts );
        
        
        %%summarize
        if opts.display_msg==1 && isfield(opts,'train_labels')
            disp([' Minibatch error: ', num2str(opts.err(1)), ' Minibatch loss: ', num2str(opts.err(2))])
        end
        
        
        opts.MiniBatchError=[opts.MiniBatchError;gather( opts.err(1))];
        opts.MiniBatchLoss=[opts.MiniBatchLoss;gather( opts.err(2))];
        
       
         %%%%%%%%%%%%%%%%%%%% here comes to the updating part;
         if (~isfield(opts.parameters,'iterations'))
            opts.parameters.iterations=0; 
        end
        opts.parameters.iterations=opts.parameters.iterations+1;
        
        
        %apply gradients

        [  net{1},res.Gate,opts ] = opts.parameters.learning_method( net{1},res.Gate,opts );
        [  net{2},res.Input,opts ] = opts.parameters.learning_method( net{2},res.Input,opts );
        [  net{3},res.Cell,opts ] = opts.parameters.learning_method( net{3},res.Cell,opts );  
        [  net{4},res.Fit,opts ] = opts.parameters.learning_method( net{4},res.Fit,opts );
     
        
    end
    
    opts.results.TrainEpochError=[opts.results.TrainEpochError;mean(opts.MiniBatchError(:))];
    opts.results.TrainEpochLoss=[opts.results.TrainEpochLoss;mean(opts.MiniBatchLoss(:))];
    
    if opts.RecordStats==1
        opts.results.TrainLoss=[opts.results.TrainLoss;opts.MiniBatchLoss];
        opts.results.TrainError=[opts.results.TrainError;opts.MiniBatchError]; 
    end
    
    toc;

end




