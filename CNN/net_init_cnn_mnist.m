


%
function net = net_init(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST


rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
%{
net.layers{end+1} = struct('type', 'pad', ...
                           'pad', 2) ;
  %}                     
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,20, 'single'), zeros(1,20,'single')}}, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool','K', 3, 'stride', 2, 'pad', 2) ;

%
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,20,50, 'single'),  zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;


net.layers{end+1} = struct('type', 'pool','K', 2, 'stride', 2) ;

%}
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,50,500, 'single'), zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,500,10, 'single'), zeros(1,10,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        net.layers{1,i}.momentum{1}=zeros(size(net.layers{1,i}.weights{1}));
        net.layers{1,i}.momentum{2}=zeros(size(net.layers{1,i}.weights{2}));
    end
end
%}
