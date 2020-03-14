# Cat-vs-Car-Detection
Two multi-layer neural networks with a single &amp; two hidden layers trained and tested. No library different than numpy has been used for the implementation of the NNs.

**PART 1 - ONE HIDDEN LAYER NN**

The gradient of the weights have been found as follows, 
![3](https://user-images.githubusercontent.com/48417171/76670689-fa7fac80-65a2-11ea-8bc0-5f4106f73b31.png)

By implementing this on python with 32 neuron one hidden layer NN the error accuracy and mse vs epoch numbers have been found as follows,

**Figure 1**
![1](https://user-images.githubusercontent.com/48417171/76670768-5a765300-65a3-11ea-8927-f171c21e8657.png)

**Figure2**
![2](https://user-images.githubusercontent.com/48417171/76670778-66faab80-65a3-11ea-9cb6-5ce1850ab29e.png)

Figure 1 and figure 2 are the outputs of the neural network that I have designed. For this neural network, the hyperbolic activation function has been implemented for all neurons. The performance has been tested with a different number of neurons in the hidden layer. Increasing hidden neuron number results with a lower MSE and accuracy test error. However, it also results in computational difficulty since more neuron means more weights to optimize i.e. more neuron means more epoch for learning. Therefore, as hidden layer neuron number 32 has been selected. With this number, I have achieved an accuracy error of % 5 in train and %25 in the test.
For assigning weights I have tried several different gaussian distributions. It is important that the weights have been initialized with zero mean and small values so that the network’s activation function will not overflow and give us a reasonable answer. As a batch number, I have tried several different numbers and I have ended up with “20”. When the batch increases a lot, I have seen that the performance of the network in terms of generalizing decreases. This is because large batches tend to converge to local sharp minimizers. Also, when batch size decreases it gets close to the gradient descent which has more noise than the mini-batch gradient descent. Therefore, I have selected a mini-batch in the middle with a size of 20.
As a learning rate, I have selected 0.2 since it has given better performance at the test set. I have seen that increasing step size results with not getting the local minimum near enough with smaller learning rates we can get closer to the local/global minimum. However, small learning rates also result in a need for a high number of epochs for training. Therefore, I have selected the learning rate again in the middle with the value of “0.2”.

For every epoch, the MSE and the accuracy error of the network overall decreases until a point of an epoch. At that point, the network stuck on a local minimum. Until the network stuck on a local minimum the MSE and the accuracy error does not decrease smoothly as we see in the gradient descent. This is because for this assignment I have used mini-batch gradient descent. Using small-batch gradient descent smooths and decreases the noise in the training whereas it has also allowed us to hinder the local minimums and give us a chance to reach the global minimum. From figure1 we can see that the error function does not always decrease but it fluctuates, i.e. the MSE value sometimes increases by noise which helps us to escape from the local minimum and reach the global minimum. After a point (epoch number) both the test and the train MSE and the accuracy errors do not change. This is the point where the network cannot learn more from that train set. This is an overtraining example because although the train errors decrease the test errors do not decrease after a point. At the point where the test error does not decrease (but the train test continues to decrease), we can say that the network is generalized and more training will result in the overtraining which will increase the test error a little bit more.

**Figure 3**
![4](https://user-images.githubusercontent.com/48417171/76670832-ceb0f680-65a3-11ea-89d5-c51958575ea9.png)

**Figure 4**
![5](https://user-images.githubusercontent.com/48417171/76670833-cfe22380-65a3-11ea-88b1-4741f76db9e8.png)

From the figures it can be seen that increasing neuron numbers can decrease the test MSE and the accuracy error. However, training a high number of neuron numbers need more epoch than the low number neurons, i.e. with more neurons it is needed to have more computational power and time. Increasing neuron numbers will increase the complexity of the network and therefore the network will learn more but also this learning will be slower because the number of weights that should be optimized has increased. On the other hand, a small number of neurons learn fast because there is a small number of weights to be optimized. However, also the complexity of the network is low and cannot learn lots of things at training as the high number neuron.
Therefore, it is better to select a neuron number between (Not high, not low) which gives us an adequate low MSE value. So that we can achieve our goals and use time efficiently.


**PART 2 - TWO HIDDEN LAYER NN**

**Figure 5**
![6](https://user-images.githubusercontent.com/48417171/76670890-16d01900-65a4-11ea-9d77-14de73cf8883.png)

**Figure 6**
![7](https://user-images.githubusercontent.com/48417171/76670891-1768af80-65a4-11ea-966b-6342a693f9cb.png)


For this part, I have trained a neural network with 2 hidden layers with 32 neurons each. The learning rate is equal to 0.2 and I have used 20 mini-batch for training.
From the figure5 and figure6, we can see that the increasing hidden layer number does not give a huge performance boost. The test accuracy error is still close %25 which is close to 1 hidden layer network. However, if we look at the training test accuracy, we can see a difference between 1 and 2 hidden layer networks. 2 hidden layer network's training accuracy error is below %1 whereas the 1 hidden layered neural network's training accuracy error is close to %5. This happens because the model complexity of the 2 hidden layers is higher than one hidden layer, i.e. the 2 hidden layer learns the set more than the 1 hidden layer. Furthermore, learning a set this much is an example of overtraining. We can see it with the help of test set whereas although 2 hidden layered network gives better results in training error it cannot generalize and give same results with the one hidden layered network at the test errors. In this cat vs car scenario increasing the hidden layer number, therefore, does not help us much.
Normally since the complexity of the 2 hidden layer network is higher than the complexity of the 1 hidden layer. I have expected to have more epochs to learn the set fully (converge). However, from figure5 and figure6, we can see that the 2 hidden layer network can learn the set faster than the 1 hidden layer setting parameters correctly. This happened because 2 hidden layer network does not work like the 1 hidden layer, i.e. 2 hidden layers can hold more information about the set than the 1 hidden layer set. An extra hidden layer may result in the network to achieve more information from the set at one iteration of training. Therefore 2 hidden layered networks may get faster to the local minimum than the 1 hidden layer networks as happened in my example.


**Figure 7**
![8](https://user-images.githubusercontent.com/48417171/76670892-18014600-65a4-11ea-87f7-d74a23faf63a.png)

**Figure 8**
![9](https://user-images.githubusercontent.com/48417171/76670893-18014600-65a4-11ea-9b50-66ddb2f65966.png)


I have trained the network in part d with a momentum of 0.2. If we compare figure7 and figure8 with figure 6 and figure 5 we can see that adding momentum result in increased learning speed. In this specific example the epoch number for converging reduced nearly 50 epochs. This happens because the adding momentum increases the magnitude of the gradient while going down the hill, i.e. when we are far from the local/global minimum. Also, when our network comes near to the local/global minimum adding momentum decreases the gradient and this new small gradient decreases the oscillations. Therefore, our network fits the local/global minimum faster that results in faster learning which we can also see by comparing figure5, figure6, figure7, and figure8.
