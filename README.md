# Dio

!!! Under construction (both the code and this readme) !!!

A neuromorphic-inspired node graph trained continuously in firing iterations and a probabilistic “firing” mechanism. A summer project made by a biology student. Not to be taken seriously.

## To design a model first and invent a problem for it later...

1,0,0,1,0,1 → 0,1,0
0,1,0,1,1,1 → 0,0,0
1,0,1,1,1,1 → 1,0,0
0,0,0,0,0,0 → 1,1,1
0,0,0,0,1,1 → 0,0,0
1,0,1,0,0,0 → 1,0,1
1,1,1,0,1,0 → 0,0,1
0,1,1,1,1,1 → 1,1,1

Do these bits have a pattern? Is there even a pattern? The answer is… yes and no. I made this dataset after all and the rule I made up was: if the first bit is 1 then the last 3 bits will be a random sequence associated with the next 5 input bits. If the first bit is 0, then the next 5 bits are an input for a logic condition (1 if all bits are the same, 0 if otherwise) repeated 3 times. This is an example of a difficult problem in machine learning as both arbitrary association and hard logic rules co-exist. This was what inspired me to attempt a node-map style model that can continuously remember and learn all these points and operates by instances where signals are fired and outputs are read to evaluate on average the correct bits.

## The math behind it

Note: I call the nodes in the network "neurons" but they operate somewhat differently than neurons in a regular DL network

# Model Architecture

## Shared Network → Networks → Neurons/Edges

### Neuron
A neuron is an object which holds a binary value, a trace value between 0 and 1, and a reference to its parent networks (yes, networks). It is the fundamental unit of this and serves as a simple placeholder. It’s closer to a node than an actual neuron but hey, I’m bad with names.

### Edge
An Edge object connects two neurons as a sender and destination. It also holds a value between -1 and 1 which represents the strength of the connection, which is explained below. It also contains a “bidirectional” trace value between -1 and 1.

### Network
A Network is an object which holds pointers to all its neurons, a set of hyperparameters and an adjacency matrix. The adjacency matrix is not an actual matrix of values (nor adjacency since it doesn’t follow classical graph rules, but again bad with names), but instead a set of coordinates which correspond to an Edge object pointer.

As mentioned before, a neuron can belong to multiple networks. When this happens it contains multiple parent networks and is the very same neuron in both networks (even if the edges leading to it have different coordinates in each adjacency matrix). For now only the first network’s hyperparameters affect its properties.

### SharedNetwork
The set of all networks in the model, and it also holds the pool of all neurons and edges that the Network objects point to. Every actual action that runs values through the networks and neurons and etc over time is run at this level.

### DataTerminal
A structure which points to a fixed set of neurons and holds its own values. Can be turned on and off via clamping and can be set to calibrate or not (which determines if it’s on during testing phases). This is used to both read certain neuron’s values and override them to feed data into the network.

### DatasetManager
Reads a csv file which contains info about which terminals it points to, if it’s shuffled or not and the values. It holds the current “instant” in the dataset timeline (if not shuffled) and updates the values in the data terminals.

### HyperOptimizer
Each sub-network contains one, it holds a history of what hyperparams the network had and the score it originated from. It can be updated with new hyperparam/score data and propose hyperparams based on a genetic algo to maximize score.

### ScoreManager
It calculates score based on a set of terminals and their value similarities (and weights) and also points to which networks it should be applied to. For example in this demo we have network 0 and 1 that are connected to terminals 0 and 1 that represent “input” and “output” respectively (, respectively?). In this case I calculate a score based on only terminal 1 and apply the optimization to networks 0 and 1 so both evolve based on network 1’s output.

---

# Model running during supervised episodes

These read one or more terminals to get scores and test the network’s ability to reproduce non-calibration terminal values.

The shared network runs cycles of iterations where at each one the value is transferred once. These cycles are composed of a train window, null window, test window.

| Calibration | Train window | Null window | Test window |
|-------------|--------------|-------------|-------------|
| True        | Clamped      | Unclamped   | Clamped     |
| False       | Clamped      | Unclamped   | Unclamped   |

This is not supposed to be how the model operates on the field where it’s unsupervised but this is supposed to represent interspaced stimuli with a neutral window where the model’s internal values stabilize. This also allows for scoring that we can use to determine the best hyperparams. The edges are only updated during the train window as it’s only during this one where we want plasticity. In every of a set amount of these cycles the final score is accumulated and fed to the optimizers in the networks and new hyperparams are proposed.

---

# Neuron firing

Each edge transfers the value from a sender to a destination. It however represents a semi-deterministic connection where the sign represents the discrete behavior of exciting or depressing and the absolute value represents the strength of the connection. What this means is that each network has a determinism fraction between 0 and 1. The firing is given by a random check per each trace.

At each iteration:

```markdown
neuron[j]_buffer = 0

for any neuron[i] == 1:
    if random < abs(edge[i][j]):
        neuron[j]_buffer += {1.0 or -1.0 depending on if edge[i][j] is positive or negative}
    else:
        neuron[j]_buffer += determinism * {1.0 or -1.0 depending on if edge[i][j] is positive or negative}

Afterwards, for each neuron buffer:
    if buffer > firing value:
        neuron[j]_spike = 1
    else:
        neuron[j]_spike = 0
```

Both the firing value (any real value) and determinism are hyperparams to be optimized.

---

# Neuron and edge traces

These represent how recently active these were. They’re updated at the shared network level since they’re not dependent on matrix operations and it’s easier to update the whole pool instead of tracking duplicated operations etc etc. They are also always updated regardless of the cycle phase. The update follows using a non-linear moving average that, given a decay rate, is expressed via:

$$
\x(t+1) = x(t)(1 - e^{-\text{decay}}) + \text{decay} \cdot \text{new\_val} \]
$$

## Neuron trace

$$
\T(t+1) = T(t)(1 - e^{-\text{decay}}) + \text{trace\_decay} \cdot \text{spike}
$$

## Edge trace (U)

$$
\U(t+1) = U(t)(1 - e^{-U_{\text{decay}}}) + U_{\text{decay}} \cdot T_{\text{sender}} \cdot 2(\text{spike}_{\text{destination}} - 0.5)
$$

---

# Edge update

Since each network has its own hyperparams and requires matrix operations, plasticity happens at the level of these and not the Shared Network (which is why I made it like this, but please note I’m not claiming I made the best choices both in code and in life).

The update rule the changes the actual capability of neuron-to-neuron firing is dependent on the destination neuron’s current value, the neurons traces, the U of the edge that’s being updated, and the other edges’ values in the network.

## Column Entropy

In order to promote heterogeneity between each neuron’s receiving values, entropy (E) is calculated via sum of entropy values of each edge’s value that points to each neuron. It is normalized to \([0,1]\) and is raised to the power of an entropy factor which is optimized.

## Total Contribution

The expected transferring of values given a matrix M is given by:

$$\
M + M^2 + M^3 + \ldots
$$

Which is approximately:

$$\
C = (I - M)^{-1} - I
$$

This is a new feature which hasn’t shown good results yet when included in the update.

## Update rule

$$\
\text{edge}[i][j] += \text{lr}(
\text{neuron}[j]_{\text{spike}} \cdot U[i][j]
- \text{reg} \cdot C[i][j] \cdot (E[j]^{\text{entropy\_factor}})
(\text{neuron[i]\_trace}^2 + \text{neuron[j]\_trace})
)
$$

Why that last term neuron[i]_trace² + neuron[j]_trace? Tried a lot of combinations or omissions and this seemed to return the best results.
