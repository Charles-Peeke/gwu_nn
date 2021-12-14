# George Washington Neural Network
This repository contains a neural network library built from scratch for use in Neural Network courses at
The George Washington University. It is a basic implementation that provides a number of layers, activation
functions, and loss functions for students to explore how networks work at a low level.

## [Presentation](https://docs.google.com/presentation/d/1fNGTJZuhLo-aVfG8b1N-oKhdmoQlpWOEGMhgxOXunHQ/edit?usp=sharing)

## Optimizers

>"Gradient descent is a way to minimize an objective function .. by updating the parameters in the opposite direction of the gradient of the objective function with respect to the parameters." [Ruder](https://arxiv.org/pdf/1609.04747)

Each of the Optimizers provided follow the same format
- Preprocess before Optimization
    - Calculate a new learning rate based on the parameters of a function including decay and iterations
- Optimize the Function
    - Update Parameters (mainly Weight and Bias) based on the algorithm
- Postprocess after Optimization
    - Update iterations for the algorithm to increase decay for next iteration

>Optimization Parameters:
>- ``learning_rate`` : [All] The size of the steps taken each optimization to reduce loss of the network
>- ``decay`` : [All] The amount that the learning rate is reduced each iteration to refine the optimization
>- ``momentum`` : [SGD] Feature to overcome local minimas to find the global minimum of the function and increase optimization
>- ``epsilon`` : [AdaG/RMS/Adam] Associated with calculating the RMS Error for the weights and bias
>- ``beta`` : [RMS/Adam] Added in various ways to help refine RMS and Adam optimization calculations

<img src="https://i.imgur.com/2dKCQHh.gif" alt="Optimization Visualization" style="height: 480px; width:620px;"/>

### SGD (Stochastic Gradient Descent)
* Amari, S.-ichi. (1993). Backpropagation and stochastic gradient descent method. Neurocomputing, 5(4-5), 185–196. https://doi.org/10.1016/0925-2312(93)90006-o | [PDF](https://bsi-ni.brain.riken.jp/database/file/141/142.pdf)
* Qian, N. (1999). On the momentum term in gradient descent learning algorithms. Neural Networks, 12(1), 145–151. https://doi.org/10.1016/s0893-6080(98)00116-6 | [PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf)

### Adagrad (Adaptive Gradient Algorithm)
* John Duchi, Elad Hazan, & Yoram Singer. (2010). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. | [PDF](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

### RMSProp (Root Mean Square Propagation)
* Kochnderfer, M. J., & Wheeler, T. A. (2019). Algorithms for optimization. The MIT Press. | [PDF](https://mitpress.mit.edu/books/algorithms-optimization)
### Adam
* Diederik P. Kingma, & Jimmy Ba. (2017). Adam: A Method for Stochastic Optimization. | [PDF](https://arxiv.org/pdf/1412.6980.pdf)

### Resources
* Zhang, A., Lipton, Z., Li, M., & Smola, A. (2021). Dive into Deep Learning. arXiv preprint arXiv:2106.11342. [Website](https://d2l.ai/index.html)
* Kinsley, H., & Kukieła, D. (2020). Neural networks from scratch in Python: Building neural networks in raw Python. Verlag nicht ermittelbar. | [Website](https://nnfs.io/)