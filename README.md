# olfactory_learning

This is a readme file regarding the simulation codes for the manuscript entitled

	"Rapid Bayesian learning in the mammlian olfactory system"
	by Naoki Hiratani, and Peter E. Latham

Questions on the manuscript and the codes should be addressed to Naoki Hiratani (N.Hiratani@gmail.com).

"bayesian_learning.py" is the main simulation code corresponding to the results depicted in Figure 3, as well as the results for "the proposed model" in Figure 4-6.
"invariant_learning.py" is an extention of "bayesian_learning.py" in which a circuit for acquisition of concentration-invariant representation is added. 

All other simulation codes are also available upon reasonable request.
Please see the maniscript for the derivations and further details.

In "bayesian_learning.py", inputs are

* coM: the average number of odors simultaneously presented. Throughout the manuscript, coM = 3 unless otherwise stated.
* M: the total nuber of the odor. We used M = 100, except for Fig. 3 and Fig. 4C.
* N: the total number of glomeruli. N = 400 throughout the manuscript.
* sigmax: the standard deviation of input Gaussian noise (sigmax = 1.0). 
* Zrho_init: a constant that detemines the initial value of the weight precision parameter (Zrho_init = 0.5, except for Fig. 6BD where Zrho_init = 0.3).
* ik: id of simulation. Curves in the figures are mean over 10 independent simulation unless otherwise stated.

And the output file "bayesian_learning_readout..." contains the odor estimation performance and the weight error after each trial.


Similarly, in "invariant_learning.py, inputs are

* coM: the average number of odors simultaneously presented (coM = 3).
* M : the total number of odor (M = 50).
* N : the total number of glomeruli (N = 200).
* sigmax : the standard deviation of input Gaussian noise (sigmax = 1.0).
* Zrho_init : a constant that determines the initial value of the weight precision (Zrho_init = 0.5)
* circuit_type: "circuit_type = 0" corresponds to the model without piriform to granule connection depicted in Fig. 7A(iii), while "circuit_type = 1" corresponds to the model depicted in Fig. 7A(iv).
* ik : id of simulation.

The output file "invariant_learning_readout..." contains the odor estimation performance and the weight error of both granule cells and piriform neurons after each trial.
