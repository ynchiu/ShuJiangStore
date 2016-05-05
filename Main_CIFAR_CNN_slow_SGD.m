clear all;


n_epoch=50;
dataset_name='cifar';
network_name='slow-cnn';
use_gpu=1;

opts.LoadResults=1;

if opts.LoadResults==1        
    files={};
    display_names={};
    max_ep=50;
    max_iter=100*max_ep; 
    TrainEpochError=[];
    TestEpochError=[];
    TrainEpochLoss=[];
    TestEpochLoss=[];
    TrainLoss=[];
    LearningRateCurves=[];
end




PrepareDataFunc=@PrepareData_CIFAR_CNN;
NetInit=@net_init_cifar_slow;

use_selective_sgd=1;
learning_method=@sgd;
sgd_lr=0.1;
ssgd_search_freq=20;
selection_reset_freq=0;

Main_Template();

if opts.LoadResults==1     
    
    display_names{end+1}='Selective-SGD freq: 20';
    files{end+1}=fullfile(opts.output_dir,opts.output_name);
    load(files{end}); 
    %LearningRateCurves=[LearningRateCurves,results.lrs(1:max_iter)];
    TrainLoss=[TrainLoss,results.TrainLoss(1:max_iter)]; 
    TrainEpochError=[TrainEpochError,results.TrainEpochError(1:max_ep)];
    TrainEpochLoss=[TrainEpochLoss,results.TrainEpochLoss(1:max_ep)];
    TestEpochError=[TestEpochError,results.TestEpochError(1:max_ep)];
    TestEpochLoss=[TestEpochLoss,results.TestEpochLoss(1:max_ep)];
end





sgd_lr=0.01;
use_selective_sgd=0;
learning_method=@sgd;
%ssgd_search_freq=10;
%selection_reset_freq=3;
Main_Template();


if opts.LoadResults==1     
    
    display_names{end+1}='SGD lr 0.01';
    files{end+1}=fullfile(opts.output_dir,opts.output_name);
    load(files{end}); 
    %LearningRateCurves=[LearningRateCurves,results.lrs(1:max_iter)];
    TrainLoss=[TrainLoss,results.TrainLoss(1:max_iter)]; 
    TrainEpochError=[TrainEpochError,results.TrainEpochError(1:max_ep)];
    TrainEpochLoss=[TrainEpochLoss,results.TrainEpochLoss(1:max_ep)];
    TestEpochError=[TestEpochError,results.TestEpochError(1:max_ep)];
    TestEpochLoss=[TestEpochLoss,results.TestEpochLoss(1:max_ep)];
end




sgd_lr=0.1;
use_selective_sgd=0;
learning_method=@sgd;
%ssgd_search_freq=10;
%selection_reset_freq=3;
Main_Template();


if opts.LoadResults==1     
    
    display_names{end+1}='SGD lr 0.1';
    files{end+1}=fullfile(opts.output_dir,opts.output_name);
    load(files{end}); 
    %LearningRateCurves=[LearningRateCurves,results.lrs(1:max_iter)];
    TrainLoss=[TrainLoss,results.TrainLoss(1:max_iter)]; 
    TrainEpochError=[TrainEpochError,results.TrainEpochError(1:max_ep)];
    TrainEpochLoss=[TrainEpochLoss,results.TrainEpochLoss(1:max_ep)];
    TestEpochError=[TestEpochError,results.TestEpochError(1:max_ep)];
    TestEpochLoss=[TestEpochLoss,results.TestEpochLoss(1:max_ep)];
end






n_epoch=50;
dataset_name='cifar';
network_name='slow-cnn';
use_gpu=1;


PrepareDataFunc=@PrepareData_CIFAR_CNN;
NetInit=@net_init_cifar_slow;

use_selective_sgd=1;
learning_method=@rmsprop;
sgd_lr=0.1;
ssgd_search_freq=20;
selection_reset_freq=0;

Main_Template();

if opts.LoadResults==1     
    
    display_names{end+1}='Selective-RMSProp freq: 20';
    files{end+1}=fullfile(opts.output_dir,opts.output_name);
    load(files{end}); 
    %LearningRateCurves=[LearningRateCurves,results.lrs(1:max_iter)];
    TrainLoss=[TrainLoss,results.TrainLoss(1:max_iter)]; 
    TrainEpochError=[TrainEpochError,results.TrainEpochError(1:max_ep)];
    TrainEpochLoss=[TrainEpochLoss,results.TrainEpochLoss(1:max_ep)];
    TestEpochError=[TestEpochError,results.TestEpochError(1:max_ep)];
    TestEpochLoss=[TestEpochLoss,results.TestEpochLoss(1:max_ep)];
end






if opts.LoadResults==1
    
   
    Data=[TrainEpochError];
    figure1=figure;
    axes1 = axes('Parent',figure1);
    hold(axes1,'on');
    plot1 = plot(Data,'Parent',axes1);
    for i=1:length(display_names)
        set(plot1(i),'DisplayName',[display_names{i},' (train)']);
    end    
    hold on;
    ax = gca;
    ax.ColorOrderIndex = 1;
    
    Data=[TestEpochError];
    plot1 = plot(Data,'--','Parent',axes1);
    for i=1:length(display_names)
        set(plot1(i),'DisplayName',[display_names{i},' (test)']);
    end
    % Create xlabel
    xlabel('Epochs','FontSize',11);
    % Create title
    title('Error Rates','FontSize',11);
    % Create ylabel
    ylabel('Error Rate','FontSize',11);
    box(axes1,'on');
    % Create legend
    legend(axes1,'show');
    
    
    
    
    
    Data=[TrainEpochLoss];
    figure1=figure;
    axes1 = axes('Parent',figure1);
    hold(axes1,'on');
    plot1 = plot(Data,'Parent',axes1);
    for i=1:length(display_names)
        set(plot1(i),'DisplayName',[display_names{i},' (train)']);
    end
    hold on;
    ax = gca;
    ax.ColorOrderIndex = 1;
    
    Data=[TestEpochLoss];
    plot1 = plot(Data,'--','Parent',axes1);
    for i=1:length(display_names)
        set(plot1(i),'DisplayName',[display_names{i},' (test)']);
    end
    
    % Create xlabel
    xlabel('Epochs','FontSize',11);
    % Create title
    title('Loss ','FontSize',11);
    % Create ylabel
    ylabel('Loss','FontSize',11);
    box(axes1,'on');
    % Create legend
    legend(axes1,'show');
    
    

end
