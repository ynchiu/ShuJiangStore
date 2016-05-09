Summary of the functions in LightNet

adagrad: implementation of the Adagrad algorithm.
adam: implementation of the Adam algorithm.
fast_conv_layer: implementation of the convolution layer.
fast_mlp_layer: implementation of linear perceptron layer.
generate_output_filename: generate output filename based on the current parameter settings.
im2col_ln: customized im2col function used in the pooling layer.
Main_Template: a template script used to train CNN and MLP networks.
maxpool: implementation of the max-pooling layer.
net_bp: implementation of the back propagation process which is used in CNN and MLP networks.
net_ff: implementation of the feed forward process which is used in CNN and MLP networks.
pad_data: a padding layer which is used in CNN.
relu: implementation of the rectified linear unit function.
rmsprop: implementation of the RMSProp algorithm.
select_learning_rate: implementation of the Selective-SGD algorithm that automatically selects the optimal learning rate at the beginning or in the middle of the training.
sgd: implementation of the stochastic gradient descent algorithm with momentum.
sigmoid_ln: implementation of the sigmoid layer.
tanh_ln: implementation of the tanh layer.
test_net: running the network in the testing mode to evaluate the current parameters.
train_net: running the network in the training mode to evaluate and calculate the loss and gradients.
TrainingScript: a training template for CNN and MLP networks.
vl_nnsoftmax: softmax layer borrowed from MatConvNet.
vl_nnsoftmaxloss: softmax log loss layer borrowed from MatConvNet.
vl_simplenn_move: a switch function between CPU and GPU adapted from MatConvNet.