clear all;
addpath('../CoreModules');
n_epoch=50;
dataset_name='cifar';
network_name='slow-cnn';
use_gpu=1; %use gpu or not 

%function handle to prepare your data
PrepareDataFunc=@PrepareData_CIFAR_CNN;
%function handle to initialize the network
NetInit=@net_init_cifar_slow;

%automatically select learning rates
use_selective_sgd=1;
%select a new learning rate every n epochs
ssgd_search_freq=20;

learning_method=@sgd;%training method: @sgd,@adagrad,@rmsprop,@adam

%sgd_lr=0.1;

Main_Template(); %call training template
