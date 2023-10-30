# **Mixes** - repository of mixture models

**The repository is available as package in PyPI: [pypi.org/project/mixes](https://pypi.org/project/mixes/).**

This repository was created as part of Research paper "Estimation of Gaussian Mixture Networks"
required as part of Master of Mathematics degree in Computational Mathematics at the University of Waterloo authored
by Andriy Dmytruk and supervised by Ryan Browne.

The repository includes implementation of the following mixture models:
* Gaussian Mixture Model ([GMM](mixes/GMM.py))
* Skew Gaussian Mixture Model ([Skew GMM](mixes/SkewGMM.py))
* Deep Gaussian Misture Modle ([DGMM](mixes/DGMM.py))
* Gaussian Mixture Network ([GMN](mixes/GMN.py))

## Usage

You can install the package [mixes](https://pypi.org/project/mixes/) from PyPI:
`pip install mixes` and use it directly, e.g.: `from mixes import Evaluator, GMM, GMN`.

The implementation is present in the [mixes/](mixes/) folder.
You can see an example of usage in the [experiments/example.ipynb](experiments/example.ipynb) jupyter notebook.

All the experiments that were performed as part of the research paper can also be found inside the [experiments/](experiments/) folder.

Kindly provide a citation to the research paper if this work proves useful to you:
```bib
@thesis{GMN,
	title        = {Estimation of Gaussian Mixture Networks},
	author       = {Andriy Dmytruk},
	year         = 2022,
	type         = {mathesis},
	institution  = {University of Waterloo}
}
```

## Models description

### Skew Gaussian Mixture Model

Skew GMM was implemented based on paper 
["Maximum likelihood estimation for multivariate skew normal mixture models"](https://www.sciencedirect.com/science/article/pii/S0047259X08001152)
by Tsung I. Lin (2006).

### Gaussian Mixture Network

GMN was proposed in the author's research paper. The model creates a network of gaussian distributions
where next layers in the model have conditional probability distribution based on the previous layer.
Each layer is a mixture of components, therefore the whole model creates a network of gaussian nodes.

The most important parameters are: 
* `layer_sizes` - these are the sizes of layers. The first layer will be used for clusterization
and therefore its size should correspond to the desired number of clusters.
* `layer_dims` - the input dimensions of each layer. Each layer has an input and output dimension. 
The output of the first layer is automatically set to the dimensionality of the data, and output of prevous layer
is considered the input of the next one. By reducing dimensionality of the deeper layers have fewer parameters,
and each layer becomes similar to Mixtures of Factor Analyzers. You would probably want to set the 
dimensions in a non-increasing order.
* `init` - determines how the model is initialized. Use `kmeans` (default) for initialization by `K-Means` and 
factor analysis on each layer. Use `random` for a completely random initialization.

The paper can be found on [Research Gate](https://www.researchgate.net/publication/365889016_Estimation_of_Gaussian_Mixture_Networks) or the [University of Waterloo website](https://uwaterloo.ca/computational-mathematics/sites/ca.computational-mathematics/files/uploads/files/dmytruk_andriy.pdf).

### Deep Gaussian Mixture Model

DGMM is based on papers ["Deep Gaussian mixture models"](https://link.springer.com/article/10.1007/s11222-017-9793-z) 
by Cinzia Viroli, Geoffrey J. McLachlan (2019) and
["Factoring Variations in Natural Images with Deep Gaussian Mixture Models"](https://proceedings.neurips.cc/paper/2014/hash/8c3039bd5842dca3d944faab91447818-Abstract.html)
by Aaron van den Oord, Benjamin Schrauwen (2014).

The parameters are similar to GMN model, as is the implementation in this repository.

The difference between DGMM and GMN is that GMN gives probabilities to layer's components conditional
on the previous layer, while DGMM has them independent.

### Annealing

We implemented deterministic annealing for mixture models as described in the paper
["On the Bumpy Road to the Dominant Mode"](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9469.2009.00681.x?casa_token=ntehyQT23A0AAAAA:pHs1_s24ZAQvg36cwjxJTcAqgH4QW-VHwOq2p-wyHNCNSeymbOR9xEdp30sfbmjI-jxdeqrvaWr6mr8)
by Hua Zhou, Kenneth L. Lange (2010).

Since the log-likelihood functions is frequently non-concave, the EM algorithm can end up
in suboptimal modes. The idea of annealing is to flatten the objective function and therefore 
increase the chances of terminating in a dominant mode. 

The parameter `use_annealing` determines whether to use annealing, while the parameter 
`annealing_start_v` determines the intial value for annealing. The value must be between `0` and `1`.
Lower values correspond to a more flattened objective function, while `1` corresponds to no
annealing. Starting for the `annealing_start_v`, the annealing value will be increased to `1` during model fitting
if `use_annealing` is set to true.

### Regularization

GMM, GMN and DGMM models have the variance regularization parameter `var_regularization`. 
Regularization makes the covariances larger on each step. This keeps the covariance matrix from becoming
close to singular, which would greatly degrade optimization for it. The parameter can also be used
for restricting the model to larger covariances and avoid overfitting.

## Stopping criterion

Use the `stopping_criterion` parameter in models to specify a stopping criterion. Specified function
must have the same signature as functions in the [mixes/stopping_criterion.py](mixes/stopping_criterion.py) file.

