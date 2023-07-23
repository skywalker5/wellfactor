# WellFactor

WellFactor is a Python implementation of Non-negative Matrix Factorization (NMF) algorithms, specifically designed for handling incomplete data. It's a core component used within the [kpd-gatech-collaboration](https://github.kp.org/CSIT-CDO-KPWA/kpd-gatech-collaboration) repository and offers a flexible interface for applying NMF to large datasets.

## Features

- Support for both dense and sparse matrices
- Option to handle incomplete observations in the data
- Customizable number of iterations and time limits
- Verbose levels for debugging and tracking progress
- Support for initialization with custom values

## Installation

You can install WellFactor using pip:
```
pip install git+https://github.kp.org/CSIT-CDO-KPWA/wellfactor.git
```

## Uninstallation

If you wish to uninstall WellFactor, you can do so with pip:
```
pip uninstall wellfactor
```
Please note that uninstalling the library will remove all its files and dependencies from your environment.

## Data Input and Output

WellFactor is designed to work with data in the form of a matrix X, where each column represents a user and rows are the features of these users. This matrix is then used as an input to the NMF algorithms.

The primary outputs of WellFactor are two matrices, W and H, that satisfy the matrix factorization equation X ≈ WH. Here, W contains the cluster representatives, and H contains the user profiles.

The output matrix H can serve as a patient profile, useful for downstream models. This utilization is more extensively discussed in the [kpd-gatech-collaboration](https://github.kp.org/CSIT-CDO-KPWA/kpd-gatech-collaboration) repository.


## Usage

Here's a simple example of how to use the `PartialObservationNMF` class:

```python
from wellfactor.nmf_partial_observation import PartialObservationNMF
import numpy as np

# Initialize a random matrix
X = np.random.random((100,200))

# Set some entries to 0
fully_observed_feature_num = 40
X[fully_observed_feature_num:, list(range(10,50))] = 0

# Run the algorithm
model = PartialObservationNMF()
W, H, _ = model.run(X, 30, fully_observed_feature_num=fully_observed_feature_num, observed_idx=[0,3], verbose=2)
```