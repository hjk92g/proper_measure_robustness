# proper_measure_robustness
Codes for the paper [Proper Measure for Adversarial Robustness](https://arxiv.org/abs/2005.02540)

## `2D_1nn_ensemble.py`: plot speculated optimally robust classifiers when data contain input noise. 
* By changing `p_ord` (default: 1), it is possible to choose different distance metric (only <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;l_p" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;l_p" title="\small l_p" /></a> norms with <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;p=1,2,&space;\infty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\small&space;p=1,2,&space;\infty" title="\small p=1,2, \infty" /></a>).
* Different examples are available by uncommenting and commenting the data definition part.
* By changing `swc_gradual` (default: 0),  it is possible to plot results using gradual nearest neighbor (1-NN) classifiers. 
* `n_noise` decides the number of classifiers will be used for getting ensemble classifiers. Using large `n_noise` will give smooth ensemble classifiers, but it takes much time.



