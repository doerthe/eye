# SEE

### Specialized Explainable inference Engines

One way to make "Specialized Explainable inference Engines" is to use the  
eye --image command line option.  
There is a concrete example at [Easter Image](https://github.com/josd/fluid/tree/master/image).  

Another way to make "Specialized Explainable inference Engines" is to train  
a "transduction from observation to prediction" model using [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)  
[transformer.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).  

### Transduction from observation to prediction

Excerpt from Wikipedia [Transduction (machine_learning)](https://en.wikipedia.org/wiki/Transduction_(machine_learning)):  

In logic, statistical inference, and supervised learning, transduction or  
transductive inference is reasoning from observed, specific (training) cases  
to specific (test) cases. In contrast, induction is reasoning from observed  
training cases to general rules, which are then applied to the test cases.  
The distinction is most interesting in cases where the predictions of the  
transductive model are not achievable by any inductive model. Note that this  
is caused by transductive inference on different test sets producing mutually  
inconsistent predictions.  

### Concrete example

- [Transduction from observation to prediction for bodies](transduction_bodies/observation_prediction_bodies.ipynb).  