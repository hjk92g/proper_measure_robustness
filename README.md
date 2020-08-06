# proper_measure_robustness
Codes for the paper [Proper Measure for Adversarial Robustness](https://arxiv.org/abs/2005.02540)

## `2D_1nn_ensemble.py`: plot speculated optimally robust classifiers when data contain input noise. 
* By changing `p_ord` (default: 1), it is possible to choose different distance metric (only <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;l_p" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;l_p" title="\small l_p" /></a> norms with <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;p=1,2,&space;\infty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;p=1,2,&space;\infty" title="\small p=1,2, \infty" /></a>).
* Different examples are available by uncommenting and commenting the data definition part.
* By changing `swc_gradual` (default: 0),  it is possible to plot results using gradual nearest neighbor (1-NN) classifiers. 
* `n_noise` decides the number of classifiers will be used for getting ensemble classifiers. Using large `n_noise` will give smooth ensemble classifiers, but it takes much time.

## `2D_genuine_Proj.py`: project points for calculation of genuine adversarial accuracy by maximum perturbation norm





### Visualization of projection process with different distance metrics (after <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;Ball(x,\epsilon)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;Ball(x,\epsilon)" title="\small Ball(x,\epsilon)" /></a> projections)
| <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{1}" title="\small \l_{1}" /></a> norm | <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{2}" title="\small \l_{2}" /></a> norm | <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{\infty}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{\infty}" title="\small \l_{\infty}" /></a> norm |
| :---:        |     :---:      |   :---: |
| <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/gifs/Proj_points_l1.gif" width="290" height="290" /> | <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/gifs/Proj_points_l2.gif" width="290" height="290" /> | <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/gifs/Proj_points_linf.gif" width="290" height="290" /> |

### Visualization of projection results
| <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{1}" title="\small \l_{1}" /></a> norm | <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{2}" title="\small \l_{2}" /></a> norm | <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{\infty}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{\infty}" title="\small \l_{\infty}" /></a> norm |
| :---:       |     :---:      |  :---: |
| <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/results/Proj_points_l1.png" width="290" height="290" /> | <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/results/Proj_points_l2.png" width="290" height="290" /> | <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/results/Proj_points_linf.png" width="290" height="290" /> |

### Visualization of simple case of properly applied adversarial training
Properly applied adversarial training refers to adversarial training with no conflicting regions originating from overlapping regions (of different classes).
| <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{1}" title="\small \l_{1}" /></a> norm | <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{2}" title="\small \l_{2}" /></a> norm | <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;\l_{\infty}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;\l_{\infty}" title="\small \l_{\infty}" /></a> norm |
| :---:       |     :---:      |  :---: |
| <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/results/Proper_adversarial_l1.png" width="290" height="290" /> | <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/results/Proper_adversarial_l2.png" width="290" height="290" /> | <img src="https://github.com/hjk92g/proper_measure_robustness/blob/master/results/Proper_adversarial_linf.png" width="290" height="290" /> |
