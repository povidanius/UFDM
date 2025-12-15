# Uniform Fourier Dependence Measure
![UFDM](ufdm.png "UFDM")

Code repository for **Uniform Fourier Dependence Measure (UFDM)**.

---

### 🛠 Build and Run with Docker

**Build the Docker image and run:**
```
docker build -t ufdm .
bash run_docker.sh
cd applications
```

## 🔬 Experiments from the Paper

### Independence tests (permutation‑based)
`applications/independence_test.py`  
Compares **UFDM** with **DCOR**, **HSIC**, and **MEF**.

Example:
```
python ./independence_test.py 750 25 
```

### Feature extraction (linear representation learning)
`applications/feature_extraction_dim_selection.py`  
Compares **UFDM**, **DCOR**, **HSIC**, **MEF**, and **Neighborhood Component Analysis (NCA)** for classification tasks.

Example:
```
./run_feature_extraction USPS
```
(will run with the orthogonality regulariser 1)

```
./run_feature_extraction_lambda_grid USPS
```
(will select the orthogonality regulariser from a grid, 3x slower)

> **Note:** Datasets for the feature‑extraction experiments are downloaded automatically via the **OpenML API**.

### Post-publication notes

We conduced some permutation tests with larger d>=50 with the same n = 750, and would like to add some practical notes.

- **UFDM estimator is sensitive to AdamW weight decay (WD).** Estimator effectiveness depend strongly on WD.
- **Permutation tests:** WD = 3.0 (used in the paper) is stable for low–moderate dimensions (d=5,15,25 as in paper); small WD (≈0.1–0.3) can substantially reduce power (e.g. for copula-based dependencies).  At d = 50, larger decay (**WD ≈ 4.5**) improves UFDM’s power.
- **For d > 50 with n = 750**, UFDM’s power degrades markedly, indicating a sample-size–to-dimension limitation.
- **Overall:** UFDM is more data-hungry than DCOR, HSIC, and MEF, and is best suited for (n,d) regimes as in paper's experiments.

  
**Feature extraction:** smaller decay (**WD = 0.1** - as in published experiments) performs better; large WD tends to over-regularise informative directions.


## Citation
```
@Article{e27121254,
AUTHOR = {Daniušis, Povilas and Juneja, Shubham and Kuzma, Lukas and Marcinkevičius, Virginijus},
TITLE = {Measuring Statistical Dependence via Characteristic Function IPM},
JOURNAL = {Entropy},
VOLUME = {27},
YEAR = {2025},
NUMBER = {12},
ARTICLE-NUMBER = {1254},
URL = {https://www.mdpi.com/1099-4300/27/12/1254},
ISSN = {1099-4300},
ABSTRACT = {We study statistical dependence in the frequency domain using the integral probability metric (IPM) framework. We propose the uniform Fourier dependence measure (UFDM) defined as the uniform norm of the difference between the joint and product-marginal characteristic functions. We provide a theoretical analysis, highlighting key properties, such as invariances, monotonicity in linear dimension reduction, and a concentration bound. For the estimation of the UFDM, we propose a gradient-based algorithm with singular value decomposition (SVD) warm-up and show that this warm-up is essential for stable performance. The empirical estimator of UFDM is differentiable, and it can be integrated into modern machine learning pipelines. In experiments with synthetic and real-world data, we compare UFDM with distance correlation (DCOR), Hilbert–Schmidt independence criterion (HSIC), and matrix-based Rényi’s α-entropy functional (MEF) in permutation-based statistical independence testing and supervised feature extraction. Independence test experiments showed the effectiveness of UFDM at detecting some sparse geometric dependencies in a diverse set of patterns that span different linear and nonlinear interactions, including copulas and geometric structures. In feature extraction experiments across 16 OpenML datasets, we conducted 160 pairwise comparisons: UFDM statistically significantly outperformed other baselines in 20 cases and was outperformed in 13.},
DOI = {10.3390/e27121254}
}
```


