clear all;
cd('./MLP');
Main_MNIST_MLP_RMSPROP();
cd ..

clear all;
cd('./CNN');
Main_CIFAR_CNN_slow_SGD();
cd ..

clear all;
cd('./LSTM');
Main_Char_LSTM();
cd ..

clear all;
cd('./ReinforcementLearning');
Main_Cart_Pole_Q_Network
cd ..
