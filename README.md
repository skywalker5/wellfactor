# WellFactor

WellFactor is a Python implementation of Non-negative Matrix Factorization (NMF) algorithms. The library has been built keeping in mind the specific requirements of handling incomplete data. It provides a flexible and easy-to-use interface for performing NMF on large datasets.

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

## Usage

Here's a simple example of how to use the `PartialObservationNMF` class:

```python
from wellfactor.nmf_partial_observation import PartialObservationNMF
import numpy as np

# Initialize a random matrix
B = np.random.random((100,200))

# Set some entries to 0
fully_observed_feature_num = 40
B[fully_observed_feature_num:, list(range(10,50))] = 0

# Run the algorithm
model = PartialObservationNMF()
W, H, _ = model.run(B, 30, fully_observed_feature_num=fully_observed_feature_num, observed_idx=[0,3], verbose=2)
```