# Neural networks

In this repository FeedForward neural networks are implemented in the C language. The current implementation supports a variable number of layers, as well as neurons in each layer. The transfer function of each neuron can be specified from one of the three definitions of: tangsig, linear, logsig. Training is accomplished using the well-known optimization method levenberg-marquardt. The levenberg-marquardt method requires a system of equations to be solved, wich in this implementation is accomplished with LAPACK. So if you are interested in using it, you need to install lapack. For a typical usage scenario check nntest.c, otherwise, if you want to understand how it can be implemented check neural_network.c


Finally, I release this code under a MIT based license, so if you intend to use it please give me credit. 
