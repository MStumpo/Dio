# Dio

!!! Under construction (both the code and this readme) !!!

## The math behind it

Note: I call the nodes in the network "neurons" but they operate somewhat differently than neurons in a regular DL network
	
### Structure

The Network object contains $n$ neurons which are placeholders for a binary value (`true`, `false`) and an `AdjacencyMatrix` object which contains a $n*n$ matrix detailing directional relationships from any neuron (column) to any other (row). This is not a real adjacency matrix in the sense that it doesn't represent absolute weights of values transferred between neurons, but can be a mix between a probabilistic signal and a regular linear weight (which can be tuned via the `--determinism` parameter) that is then processed in neuron firing, which is discussed below.

### Timesteps

What for now are called "timesteps" are iterations where one instance of neuron firing is carried out and all the signals carried by the neurons are carried once. In the code, an epoch is defined as set of iterations where all the datapoints are fed to the network, each for a set number of iterations before the next datapoint is fed (this can be set as `--time-window`). The network is envisioned to work in a real-time-like fashion where it can be exposed to data and learn continuously.

### Neuron Firing

The `neuronFiring` method of the network is responsible for taking the binary signals in the network and moving them around with respect to the adjacency matrix. The adjacency matrix, when determinism is 0, contains values from -1 to 1 where the absolute value is treated as a probability of firing the current signal and the sign is treated as wether the signal is positive or negative (for example: if a neuron carrying 1 has a "synapse" of 0.5 to another one, the receiving neuron has a 50% chance of having its final state increased by 1, and for a value of -0.5 the receiving neuron has a 50% chance of having its final state decreased by 1).`adjMatrix[i][j]` represents the connection from the i'th neuron to the j'th. During the firing the neurons attain a temporary sum of their receiving values which are then clipped to either 0 or 1 in the end for the next timestep (according to wether or not the sum is bigger than the firing value, which is modifiable by `--firing-value` but defaults at 1.0). In this way even though neurons can only carry positive values they can be inhibitory to some and excitatory to others. The `--determinism` parameter splits the signal into a guaranteed sum of $determinism$ and a possible sum of $1 - determinism$ if a random value check passes the respective adjacency matrix value (which itself is not modified by determinism).

### Neuron history and correlation

In every timestep the current state of the neurons is stored in a `neuronHistory` object (a `vector<vector<bool>>`). It's updated via `push_back` so the larger indices represent more recent values. When updating the adjacency matrix, it's possible to calculate a $n*n$ correlation matrix from every neuron to every other in an assymetric manner based on the timing of different true values. `correlation[i][j]` represents j's correlation from i, to which is added a positive score every time j attains a true value (in timestep $t_2$) after i attains a true value (in timestep $t_1$), or a negative score when vice versa. Both scores can be calculated via:

$\LARGE e^{\frac{|\Delta t|}{\tau_+}} * decay^{size - max(t_1,t_2)}$ and $\LARGE -e^{\frac{|\Delta t|}{\tau_-}} * decay^{size - max(t_1,t_2)}$ 

The decay value is actually wrongly named as it's a conservation value to determine how much each historic firing event is depreciated based on how old it is and to decrease overfitting in continuous updates. This formula is based on spike-dependent time plasticity as it's used in various neuromorphic models. $\tau_+$, $\tau_-$ and $decay$ can be modified via `--tau-pos`, `--tau-neg` and `--decay`.

### Cross Kernel Matrix Entropy

This fancy expression literally means a $n\*n$ matrix filled with entropy values calculated from a cross-shaped kernel throughout the adjacency matrix. A sliding cross is taken and all the values in the range of `kernelSize` (modifiable via `--kernel-size`) are taken and calculated via the shannon entropy formula ( $-p*log(p)$ ) for the respective adjacency matrix element. Since adjacency matrix values can be negative and hence not real probabilities, kernel normalization ( `--kernel-normalization 0 or 1` ) can be implemented to either consider $p = |adjMatrix[i][j]|$ or $p = (adjMatrix[i][j] + 1)/2$ . The final returned value is normalized so that the matrix entropy values are between 0 and 1. These entropy values are then directly multiplied in the total update and can be used to directly control the "chaos" of the network. The rationale is the interpretation of entropy as a measure of additive relative surprise and the curve achieves a maximum between 0 and 1 (closer to 0) and is equal to 0 when $p = 1$ or 0. This allows to bias the learning towards "neat" values (-1, 0 or 1) and avoid (or promote) "disorganized" and highly probabilistic networks. The cross kernel is meant to only consider neighboring synapses such as neurons that receive from the same neuron or synapses in the same receiving neuron from other neurons.

### Adjacency matrix update

$\LARGE adjMatrix[i][j] \+\= lr\*crossCorrelation[i][j]\*reward*{entropy[i][j]}^{entropyFactor} - reg*adjMatrix[i][j]$, clipped to $[-1,1]$

At each timestep, this update is made to every member of the adjacency matrix. The reward is not used for now, but can be implemented if there is a need for a global "don't do that" or "do more of that" signal to update for the real-time correlation. The reg value, between 0 and 1, is meant to promote unstable small updates and will push `adjMatrix[i][j]` toward 0 unless the active update term is able to "overcome" it. These values can be adjusted via `--lr`, `--entropy-factor` and `--reg`.
