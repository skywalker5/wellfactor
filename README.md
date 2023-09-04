# WellFactor

WellFactor is a Python implementation of Non-negative Matrix Factorization (NMF) algorithms, specifically designed for handling incomplete data. It's a core component used within the [kpd-gatech-collaboration](https://github.kp.org/CSIT-CDO-KPWA/kpd-gatech-collaboration) repository and offers a flexible interface for applying NMF to large datasets.

## Installation

You can install WellFactor using pip:
```
pip install git+https://github.com/skywalker5/wellfactor.git
```

## Uninstallation

If you wish to uninstall WellFactor, you can do so with pip:
```
pip uninstall wellfactor
```
Please note that uninstalling the library will remove all its files and dependencies from your environment.

## Key Features

WellFactor implements two key methods: `NMF` and `PartialObservationNMF`.

- `NMF`: This is an algorithm that executes Nonnegative Matrix Factorization (NMF) using the Alternating Nonnegative Least Squares (ANLS) approach under the assumption that the data matrix is fully observable.

- `PartialObservationNMF`: This is an algorithm that can be used when parts of the data matrix are not fully observable. This is particularly useful for handling incomplete data or missing values.

Depending on the observability of your data matrix, you can choose the appropriate method that suits your needs.

## Data Input and Output

WellFactor is designed to work with data in the form of a matrix X, where each column represents a user and rows are the features of these users. For instance, X could represent a TF-IDF matrix of user's activity.

Imagine a scenario where we have 5 users and their activity is represented in terms of 3 features:

|   | User 1 | User 2 | User 3 | User 4 | User 5 |
|---|--------|--------|--------|--------|--------|
| Feature 1 |   0.1  |   0.8  |   0.3  |   0.5  |   0.2  |
| Feature 2 |   0.6  |   0.0  |   0.7  |   0.1  |   0.6  |
| Feature 3 |   0.4  |   0.2  |   0.0  |   0.4  |   0.2  |

This matrix X is then used as an input to the NMF algorithms, resulting in two matrices W and H, that satisfy the matrix factorization equation X ≈ WH^T.

Matrix W represents cluster centers in the feature space and can be viewed as the 'basis vectors' that generate the feature representation for each user. It is interpreted as the clustering from the data-driven perspective. For example, if we factorize X into 2 clusters, we might have:

|   | Cluster 1 | Cluster 2 |
|---|-----------|-----------|
| Feature 1 |    0.7    |    0.3    |
| Feature 2 |    0.2    |    0.6    |
| Feature 3 |    0.1    |    0.1    |

Matrix H represents how much each user pertains to each of the discovered clusters (in the W matrix). It can be interpreted as the weights of the 'basis vectors' for each user and provides a latent factor model of the users. It might look like:

|   | Cluster 1 | Cluster 2 |
|---|-----------|-----------|
| User 1 |   0.2  |   0.8  |
| User 2 |   0.5  |   0.5  |
| User 3 |   0.3  |   0.7  |
| User 4 |   0.7  |   0.3  |
| User 5 |   0.1  |   0.9  |

The primary outputs of WellFactor are these matrices, W and H. The output matrix H can serve as a patient profile, useful for downstream models. This utilization is more extensively discussed in the [kpd-gatech-collaboration](https://github.kp.org/CSIT-CDO-KPWA/kpd-gatech-collaboration) repository.



## Usage Examples

Here are some simple examples of how to use the `NMF` and `PartialObservationNMF` classes:

### Using the NMF Class

```python
from wellfactor.nmf import NMF
import numpy as np

# Initialize a random matrix
X = np.random.random((100,200))

# Set the number of factors
num_factors = 10

# Run the algorithm
model = NMF()
W, H = model.run(X, num_factors, verbose=2)
```

In this example, we initialize a random matrix `X` with dimensions 100 by 200, then set the number of factors we wish to factorize `X` into to 10. We then run the NMF algorithm on `X` using the `NMF` class.

### Using the PartialObservationNMF Class

```python
from wellfactor.nmf_partial_observation import PartialObservationNMF
import numpy as np

# Initialize a random matrix
X = np.random.random((100,200))

# Set some entries to 0 to simulate partial observability
fully_observed_feature_num = 40
X[fully_observed_feature_num:, list(range(10,50))] = 0

# Run the algorithm
model = PartialObservationNMF()
W, H, _ = model.run(X, 30, fully_observed_feature_num=fully_observed_feature_num, observed_idx=[0,3], verbose=2)
```

In this example, we simulate partial observability by setting some entries of the random matrix `X` to 0. We then run the PartialObservationNMF algorithm on `X` using the `PartialObservationNMF` class.

More detailed examples can be found in the examples directory.