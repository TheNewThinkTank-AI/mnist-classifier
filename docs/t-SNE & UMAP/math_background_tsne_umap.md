# Mathematical foundations of **t-SNE** and **UMAP**

## 1. t-SNE: t-Distributed Stochastic Neighbor Embedding

### High-Level Idea

t-SNE converts similarities between data points into probabilities and tries to minimize the difference between these probabilities in high-dimensional and low-dimensional spaces.

### Step-by-step math breakdown

#### Step 1: Compute pairwise similarities in high-dimensional space

We define a **conditional probability** $ p_{j|i} $ that represents the similarity of point $ x_j $ to point $ x_i $. This is based on a **Gaussian distribution centered at $ x_i $**:

$$
p_{j|i} = \frac{\exp\left(-||x_i - x_j||^2 / (2\sigma_i^2)\right)}{\sum_{k \neq i} \exp\left(-||x_i - x_k||^2 / (2\sigma_i^2)\right)}
$$

- $ \sigma_i $ is chosen per point $ i $ so that the entropy of $ p_{j|i} $ matches a user-defined **perplexity**.
- Perplexity roughly corresponds to a guess of the number of neighbors around each point.

Then, we symmetrize this into a joint probability:

$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}
$$

This gives us an $ N \times N $ matrix of similarities.

#### Step 2: Compute similarities in low-dimensional space

In the low-dimensional embedding (say 2D), we compute a similar probability $ q_{ij} $, but now using a **Student’s t-distribution with 1 degree of freedom** (which becomes a Cauchy distribution):

$$
q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}
$$

This heavy-tailed distribution helps avoid the **crowding problem** — where many points crowd together in lower dimensions when using Gaussian distributions.

#### Step 3: Minimize KL Divergence

We minimize the **Kullback-Leibler divergence** between the two distributions $ P $ and $ Q $:

$$
KL(P || Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

This cost function is minimized via **gradient descent**:

$$
\frac{\partial KL}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ||y_i - y_j||^2)^{-1}
$$

---

#### Summary of t-SNE Math:

| Step | Description |
|------|-------------|
| 1 | Compute Gaussian-based similarities in high-D |
| 2 | Compute t-distribution-based similarities in low-D |
| 3 | Minimize KL-divergence between the two distributions |

---

## 2. UMAP: Uniform Manifold Approximation and Projection

UMAP is inspired by **topology and Riemannian geometry**, assuming that data lies on a **low-dimensional manifold** embedded in high-dimensional space.

It constructs a topological representation of the data and finds a low-dimensional layout that preserves this structure.

### Step 1: Construct Fuzzy Topological Representation

UMAP builds a **fuzzy simplicial complex** over the data:

#### Local Metric Estimation:

For each point $ x_i $, UMAP estimates a local scale $ \rho_i $, which is the distance to its nearest neighbor.

Then, for each pair $ (x_i, x_j) $, it computes a fuzzy membership value:

$$
w_{ij} = \exp\left( - \frac{||x_i - x_j|| - \rho_i}{\sigma_i} \right)
$$

- $ \rho_i $: distance to nearest neighbor → captures local density
- $ \sigma_i $: scaling factor set such that the sum of weights approximates a desired local connectivity

This results in a **directed graph** where edge weights represent "fuzzy" proximity.

To make it undirected, UMAP combines forward and reverse edges:

$$
p_{ij} = w_{ij} + w_{ji} - w_{ij} \cdot w_{ji}
$$

### Step 2: Create Low-Dimensional Graph with Similar Structure

UMAP then creates a similar fuzzy simplicial complex in the low-dimensional space using a different kernel:

$$
q_{ij} = (1 + a ||y_i - y_j||^{2b})^{-1}
$$

- Parameters $ a $ and $ b $ are typically learned or set heuristically.
- The idea is to preserve the same kind of **connectivity** as in the high-dimensional space.

### Step 3: Optimize Layout Using Cross-Entropy

Instead of KL-divergence like t-SNE, UMAP minimizes **cross-entropy** between the two fuzzy graphs:

$$
CE(P, Q) = \sum_{i \neq j} \left[ p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}} \right]
$$

This loss encourages both:
- Attraction between nearby points (via $ p_{ij} \log \frac{p_{ij}}{q_{ij}} $)
- Repulsion between distant points (via $ (1 - p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}} $)

UMAP uses **stochastic gradient descent** to optimize this efficiently.

### Summary of UMAP Math:

| Step | Description |
|------|-------------|
| 1 | Build a fuzzy graph based on local distances |
| 2 | Model low-dimensional layout with a similar graph |
| 3 | Minimize cross-entropy between graphs |

## Comparison Table: t-SNE vs UMAP (Mathematically)

| Feature | t-SNE | UMAP |
|--------|-------|------|
| Goal | Preserve local neighbor relationships | Preserve both local and global structure |
| Similarity Measure | Gaussian (high-D), t-distribution (low-D) | Fuzzy sets with exponential decay |
| Optimization Objective | KL-divergence | Cross-entropy |
| Distance Preservation | Poorly preserved globally | Better preserved globally |
| Randomness | Depends on initialization | Deterministic with fixed seed |
| Speed | Slower | Faster due to graph-based optimization |

## Practical Notes on MNIST

When applied to MNIST:
- **t-SNE** tends to produce compact clusters with clear separation between digit classes, but may not show how clusters relate to each other.
- **UMAP** often shows more meaningful spacing between clusters (e.g., 0 near 6/9, 1 far from 8), and sometimes reveals substructures within clusters (e.g., variations in how people write digits).

## Coming up

- For **t-SNE**: perplexity, early exaggeration, and gradient descent steps.
- For **UMAP**: The role of Riemannian manifolds, fuzzy topological structures, and spectral graph theory.
