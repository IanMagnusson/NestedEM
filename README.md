# NestedEM
Expectation-Maximization derivation and implementation for a nested mixture of mixtures 
* Solves two-population dataset, each a different mixture of underlying distributions
* Utilized Numpy for performant implementation

Handles basic EM and the following special case. The dataset is the union of two populations drawn from the following distributions:

- <img src="https://latex.codecogs.com/svg.latex?p_X(x)%20=%20\alpha%20\mathcal{N}(\mu_1,\sigma_1^2)%20+%20(1-\alpha)\mathcal{N}(\mu_2,\sigma_2^2)" /> 

- <img src="https://latex.codecogs.com/svg.latex?p_Y(y)%20=%20\beta%20\mathcal{N}(\mu_1,\sigma_1^2)%20+%20(1-\beta)\mathcal{N}(\mu_2,\sigma_2^2)" /> 


See main.pdf problem 3 for detailed specification, problem 3.b for derivation of special case update rules, and problem 3.c for discussion of results.

Run EM.ipynb to see detailed explanation of results on various distributions and to demo implementation.
