function [ net,opts ] = selective_sgd( net,opts )
%NET_APPLY_SELECTIVE_SGD Summary of this function goes here
%   Detailed explanation goes here


    if (mod(opts.parameters.current_ep,opts.parameters.ssgd_search_freq)==1||opts.parameters.ssgd_search_freq==1)

        if ~isfield(opts.parameters,'selection_count')
            opts.parameters.selection_count=0;
        end

        if ~isfield(opts.parameters,'selection_reset_freq')
            opts.parameters.selection_reset_freq=0;
        end


        if ~isfield(opts.parameters,'init_train')
            opts.parameters.init_train=1;
        end

        
        if mod(opts.parameters.selection_count,opts.parameters.selection_reset_freq)==1 && opts.parameters.selection_count >1 && opts.parameters.selection_reset_freq>0
            lr_best=opts.parameters.selected_lr(1); %reset
        else
            [lr_best]=select_learning_rate(net,opts);         
        end
        opts.parameters.lr=lr_best;

        
        
        
        if(opts.parameters.init_train&&opts.parameters.selection_count==0)
           disp(['Initial training.']);
           opts.parameters.initial_lr=lr_best*0.1;
           opts.parameters.lr=opts.parameters.initial_lr;
           opts.parameters.current_ep=0;
           
           if isfield(opts.parameters,'asgd_lr')
               opts.parameters.asgd_lr_bak=opts.parameters.asgd_lr; 
               opts.parameters.asgd_reset_freq_bak=opts.parameters.asgd_reset_freq;
               opts.parameters.asgd_lr=1e-2;
               opts.parameters.asgd_reset_freq=0;
           end
        else
            
            if ~isfield(opts.parameters,'selected_lr')
                opts.parameters.selected_lr(1)=lr_best;
            else        
                opts.parameters.selected_lr(end+1)=lr_best;
            end
            
            if isfield(opts.parameters,'asgd_lr_bak')
               opts.parameters.asgd_lr=opts.parameters.asgd_lr_bak; 
               opts.parameters.asgd_reset_freq=opts.parameters.asgd_reset_freq_bak;
            end
           
           
        end
        
        
        opts.parameters.lr_max=opts.parameters.lr;
        opts.parameters.lr_min=opts.parameters.lr_max*1e-4;

        if isfield(opts.parameters,'asgd_lr')
            if ~isfield(opts.parameters,'asgd_lr_decay')
                opts.parameters.asgd_lr_decay=0.9;
            end
            
        end
        if strcmp(func2str(opts.parameters.learning_method),'adaptive_sgd_ew')
            for i=1:numel(net.layers)
                 if strcmp(net.layers{i}.type,'conv')||strcmp(net.layers{i}.type,'mlp')
                     if isfield(net.layers{1,i},'lr')
                         %%set new learning rates
                        net.layers{1,i}.lr{1}=ones(size(net.layers{1,i}.weights{1}))*opts.parameters.lr_max;
                        net.layers{1,i}.lr{2}=ones(size(net.layers{1,i}.weights{2}))*opts.parameters.lr_max;
                     end
                end
            end
            
            if(opts.parameters.asgd_mom>0)        
                for i=1:numel(net.layers)
                     if strcmp(net.layers{i}.type,'conv')||strcmp(net.layers{i}.type,'mlp')
                         if ~isfield(net.layers{1,i},'momentum_d')
                            net.layers{1,i}.momentum_d{1}=zeros(size(net.layers{1,i}.weights{1}));
                            net.layers{1,i}.momentum_d{2}=zeros(size(net.layers{1,i}.weights{2}));
                         end
                    end
                end    
            end
            
            net.mom_factor_d=0; 
            
            
        end
        
        opts.parameters.selection_count=opts.parameters.selection_count+1;
        
        disp(['Selected learning rate: ',num2str(opts.parameters.lr)]);

    end
        
end

