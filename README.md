# DQN


Implementation of:

  -DQN -> http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf
  
  -Double DQN -> http://arxiv.org/abs/1509.06461
  
  -Dueling network -> http://arxiv.org/abs/1511.06581
  
  -Prioritized experience replay -> http://arxiv.org/pdf/1511.05952v3.pdf
  
  -Deep Recurrent Q-Learning For Partially Observable MDPs -> https://arxiv.org/pdf/1507.06527.pdf (Soon, allow to get ride of preprocessing)
  
  
Currently, the config file parameters are tunned in order to learn to play to flappy bird (https://github.com/sourabhv/FlapPyBird).


#

Dependencies:

  -Tensorflow
  
  -python 2.7
  
  -openCV
  
  -pyQtGraph
  
  -numpy
  
  -pqdict

#

11/20/2016 - learning DDQN & Dual network OK

11/23/2016 - PER integrated but not tested yet



![alt tag](https://github.com/thbeucher/DQN/blob/master/images/figure_1.png)
![alt tag](https://github.com/thbeucher/DQN/blob/master/images/figure_2.png)
![alt tag](https://github.com/thbeucher/DQN/blob/master/images/eval50.png)
