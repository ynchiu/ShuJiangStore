n_epoch=20; %training epochs
dataset_name='mnist'; %dataset name
network_name='cnn'; %network name
use_gpu=1; %use gpu or not 

%function handle to prepare your data
PrepareDataFunc=@PrepareData_MNIST_CNN;
%function handle to initialize the network
NetInit=@net_init_cnn_mnist;

%automatically select learning rates
use_selective_sgd=1; 
%select a new learning rate every n epochs
ssgd_search_freq=10; 
learning_method=@sgd; %training method: @sgd,@adagrad,@rmsprop,@adam

%sgd parameter 
%(unnecessary if selective-sgd is used)
%sgd_lr=5e-2;

Main_Template(); %call training template