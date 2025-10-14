# Uniform Fourier Dependence Measure
![UFDM](ufdm.png "UFDM")

Code repository for **Uniform Fourier Dependence Measure (UFDM)**.

---

### 🛠 Build and Run with Docker

**Build the Docker image and run:**
```
docker build -t ufdm .
bash run_docker.sh
```

## 🔬 Experiments from the Paper

### Independence tests (permutation‑based)
`applications/independence_test.py`  
Compares **UFDM** with **DCOR**, **HSIC**, and **MEF**.

Example:
```
python ./independence_test.py 1500 5 gaussian
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
(will select the orthogonality regulariser from a grid, very slow)


> **Note:** Datasets for the feature‑extraction experiments are downloaded automatically via the **OpenML API**.

