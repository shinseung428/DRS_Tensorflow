# Discriminator Rejection Sampling Tensorflow

Tensorflow implementation of [Discriminator Rejection Sampling](https://arxiv.org/pdf/1810.06758.pdf)  

## Discriminator Rejection Sampling
![pseudocode](./images/pseudocode.png) 
![equation](./images/equation.png) 

SAGAN with hinge loss is used as a base model. The network is first trained for 50 epochs, and as it's descriped in the paper, the model is further trained using a smaller learning rate (5 epochs). The model uses hinge loss and a sigmoid output is requried to perform DRS. Therefore fc layers are added on top of the discriminator and is further trained using a cross entropy loss.

The rejection sampling algorithm is implemented in the sample.py file.  
Once finished training, about 50k samples are generated to estimate M (BurnIn phase).  
After estimating M, new samples are generated and their F_hat values are calculated.  
For CelebA dataset, 80th percentile value is used for gamma (80th percentile value of F is used as gamma). 


## Requirements
* numpy
* opencv2
* Tensorflow

## Results
### High acceptance rate
![high](./images/high.gif)

### Low acceptance rate
![low](./images/low.gif)



