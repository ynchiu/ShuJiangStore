clear all;
cd('./MLP');
disp('Test multilayer perceptron.')
Main_MNIST_MLP_RMSPROP();
cd ..

clear all;

cd('./ReinforcementLearning');
disp('Test Q-network.')
Main_Cart_Pole_Q_Network
cd ..


clear all;
cd('./LSTM');
disp('Test LSTM.')
Main_Char_LSTM();
cd ..


clear all;
cd('./CNN');
disp('Test convolutional neural network.')
Main_CIFAR_CNN_slow_SGD();
cd ..
