function [lr_best,min_cost] = select_learning_rate(net,opts )
%NET_APPLY_GRAD_SGD Summary of this function goes here
%   Detailed explanation goes here

   % do your selective sgd
   temp_net=net;
   net=[];
   opts.parameters.cost=zeros(size(opts.parameters.lrs));

    for l=1:length(opts.parameters.lrs)%learning rate list

        net=temp_net;
        opts.parameters.lr=opts.parameters.lrs(l);%test the candidate learning rate
        
        if strcmp(func2str(opts.parameters.learning_method),'adaptive_sgd_ew')
            
            opts.parameters.lr_max=opts.parameters.lr;
            
            for i=1:numel(net.layers)
                 if strcmp(net.layers{i}.type,'conv')||strcmp(net.layers{i}.type,'mlp')
                     if ~isfield(net.layers{1,i},'lr')
                        net.layers{1,i}.lr{1}=ones(size(net.layers{1,i}.weights{1}))*opts.parameters.lr_max;
                        net.layers{1,i}.lr{2}=ones(size(net.layers{1,i}.weights{2}))*opts.parameters.lr_max;
                     end
                end
            end
            
        end
        

        if (isfield(opts.parameters,'selected_lr') && length(opts.parameters.selected_lr)>0 && opts.parameters.lr>opts.parameters.selected_lr(1)) 

            cost=100;
            opts.parameters.cost(l) =cost;
            continue;


        else

            for mini_b=1:min(opts.n_batch,opts.parameters.search_iterations)

                idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);
                if length(size(opts.train))==2%%train mlp                    
                    res(1).x=opts.train(:,idx);
                else %train cnn

                    res(1).x=opts.train(:,:,:,idx);
                end
                res(1).class=opts.train_labels(idx);

                %forward
                [ net,res,opts ] = net_ff( net,res,opts );

                %%%%backward
                dzdy=single(1.0);
                opts.dzdy=dzdy;
                if opts.use_gpu
                    opts.dzdy=gpuArray(dzdy);
                end
                [ net,res,opts ] = net_bp( net,res,opts );


                %%collect stats

                %err=error_multiclass(res(1).class,res);
                %err=gather(err);

                cost=gather(res(end).x/opts.parameters.batch_size);

                if cost>100
                   cost=100;
                   break; 
                end

                [net,res,opts] = opts.parameters.learning_method(net,res,opts);

            end

        end

        opts.parameters.cost(l) =cost;
        disp(['Learning rate: ',num2str(opts.parameters.lrs(l)),' cost: ' num2str(cost)]);

    end

    [min_cost,min_idx]=min(opts.parameters.cost);


    lr_best=opts.parameters.lrs(min_idx);
    
                
end

