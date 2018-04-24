# Transduction

### Introduction

Excerpt from [https://en.wikipedia.org/wiki/Transduction_(machine_learning)](https://en.wikipedia.org/wiki/Transduction_(machine_learning)):  

In logic, statistical inference, and supervised learning, transduction or transductive inference is reasoning from observed, specific (training) cases to specific (test) cases. In contrast, induction is reasoning from observed training cases to general rules, which are then applied to the test cases. The distinction is most interesting in cases where the predictions of the transductive model are not achievable by any inductive model. Note that this is caused by transductive inference on different test sets producing mutually inconsistent predictions.  

### Concrete example: Transduction from observation to prediction

It requires [Python3](https://www.python.org/), [tensorflow](https://pypi.org/project/tensorflow/) and [tensor2tensor](https://pypi.org/project/tensor2tensor/).  

See the [observation_prediction problem](https://github.com/josd/eye/blob/master/transduction/observation_prediction.py).  

Run the [observation_prediction script](https://github.com/josd/eye/blob/master/transduction/observation_prediction.sh).  

See the transductions from [sample.observation](https://github.com/josd/eye/blob/master/transduction/sample.observation) to [sample.prediction](https://github.com/josd/eye/blob/master/transduction/sample.prediction):  

```
A_PERSON with weight 74 kg and height 179 cm
A_TURBINE with size factor 4 and subjected to windspeed 62 km/h
->-
A_PERSON has BMI class N	-0.37	A_PERSON has BMI class U	-4.58	A_PERSON has BMI class O	-4.83
A_TURBINE producing 9533 kW	-0.37	A_TURBINE producing 16682 kW	-3.19	A_TURBINE producing 2383 kW	-3.54
```
