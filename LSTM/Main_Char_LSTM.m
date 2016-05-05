clear all;

%%%%%%%%%%%%%This example will need to be reorganized


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%provide parameters and inputs below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../')

addpath('../CoreModules');

addpath('./lm_data');

n_epoch=15; %%training epochs
dataset_name='char'; % dataset name
network_name='lstm';
use_gpu=0; %%use gpu or not 

PrepareDataFunc=@PrepareData_Char_LSTM; %%function handler to prepare your data
NetInit=@net_init_char_lstm;  %% function to initialize the network


use_selective_sgd=0; %automatically select learning rates
%%selective-sgd parameters
%ssgd_search_freq=10; %select new coarse-scale learning rates every n epochs


learning_method=@rmsprop; %training method: @sgd,@adaptive_sgd_ew;

%sgd parameter (unnecessary if selective-sgd is used)
sgd_lr=1e-1;




opts.parameters.batch_size=100;
opts.parameters.n_hidden_nodes=30;
opts.parameters.n_hidden_layer_nodes=100;
opts.parameters.n_cell_nodes=30;
opts.parameters.n_input_nodes=67;
opts.parameters.n_output_nodes=67;
opts.parameters.n_gates=3;
opts.parameters.n_frames=64;%%%%sentence length, may need to change in each call?

opts.parameters.lr =sgd_lr;
opts.parameters.mom =0.9;
opts.parameters.learning_method=learning_method;
opts.parameters.selective_sgd=use_selective_sgd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%provide parameters and inputs above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stupid settings below (so please ignore)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.n_epoch=n_epoch; %training epochs
opts.dataset_name=dataset_name; %dataset name
opts.network_name=network_name; %network name
opts.use_gpu=use_gpu; %use gpu or not 

opts.results=[];
opts.results.TrainEpochError=[];
opts.results.TestEpochError=[];
opts.results.TrainEpochLoss=[];
opts.results.TestEpochLoss=[];
opts.RecordStats=1;
opts.results.TrainLoss=[];
opts.results.TrainError=[];

opts.plot=1;

opts.dataDir=['./',opts.dataset_name,'/'];
opts=PrepareDataFunc(opts);

net=NetInit(opts);


opts=generate_output_filename(opts);

if(opts.use_gpu)       
    for i=1:length(net)
        net(i)=vl_simplenn_move(net(i),'gpu');
    end
else
    for i=1:length(net)
        net(i)=vl_simplenn_move(net(i),'cpu');
    end
end


opts.n_batch=floor(opts.n_train/opts.parameters.batch_size);
opts.n_test_batch=floor(opts.n_test/opts.parameters.batch_size);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stupid settings above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training goes below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


opts.parameters.current_ep=1;

start_ep=opts.parameters.current_ep;
if opts.plot
    figure1=figure;
end
for ep=start_ep:opts.n_epoch
    
    
    [net,opts]=train_lstm(net,opts);  
    [opts]=test_lstm(net,opts);
    opts.parameters.current_ep=opts.parameters.current_ep+1;
    disp(['Epoch ',num2str(ep),' testing error: ',num2str(opts.results.TestEpochError(end)), ' testing loss: ',num2str(opts.results.TestEpochLoss(end))])
    
    %
    if opts.plot
        subplot(1,2,1); plot(opts.results.TrainEpochError);hold on;plot(opts.results.TestEpochError);hold off;title('Error Rate per Epoch')
        subplot(1,2,2);plot(opts.results.TrainEpochLoss);hold on;plot(opts.results.TestEpochLoss);hold off;title('Loss per Epoch')
        drawnow;
    end
    %}
    parameters=opts.parameters;    
    results=opts.results;
    save([fullfile(opts.output_dir,opts.output_name)],'net','parameters','results');     
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training goes above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


