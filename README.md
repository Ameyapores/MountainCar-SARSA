# MountainCar-SARSA
Python Implementation of discrete and Radial basis function SARSA on mountaincar environment
1) Discretization: 65 discrete static states.
2) Radial basis approx: 64 basis given by linear combination of 64 guassians.

## Results
### Discretized SARSA
*Plot of rewards vs number of episodes*


<img align="left" img src="images/Figure_1.png" width="400"> <br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>   
*Plot of Cost-to-go function vs Postion, Velocity*

<figure>
  <img src="images/animated_volcano0.gif" width="400"> 
  <figcaption align="center"> Fig 1: Episode 0 </figcaption>
  <img src="images/animated_volcano12.gif" width="400" >
  <figcaption> Fig 2: Episode 12 </figcaption> 
  <img src="images/animated_volcano104.gif" width="400">
  <figcaption> Fig 3: Episode 100 </figcaption>
  <img src="images/animated_volcano.gif" width="400">
  <figcaption> Fig 4: Episode 1000 </figcaption>
<figure>

### Radial Basis function SARSA
*Plot of rewards vs number of episodes*

<img align="left" img src="images/Figure_2.png" width="400"> <br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

*Plot of Cost-to-go function vs Postion, Velocity*

<figure>
  <img src="images/AV0.gif" width="400"> 
  <figcaption> Fig 1: Episode 0 </figcaption>
  <img src="images/AV250.gif" width="400" >
  <figcaption> Fig 2: Episode 250 </figcaption> 
  <img src="images/AV1000.gif" width="400">
  <figcaption> Fig 3: Episode 1000 </figcaption>
  <img src="images/AV.gif" width="400">
  <figcaption> Fig 4: Episode 2000 </figcaption>
<figure>
  
Sources:
1) Sutton, R.S., 1996. Generalization in reinforcement learning: Successful examples using sparse coarse coding. In Advances in neural information processing systems (pp. 1038-1044).
2) Sutton, R.S. and Barto, A.G., 1998. Introduction to reinforcement learning (Vol. 135). Cambridge: MIT press.
