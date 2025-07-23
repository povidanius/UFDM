# UFDM
![UFDM](ufdm.png "UFDM")

Code repository for **Uniform Fourier Dependence Measure (UFDM)**.

---

### 📄 Related Publication

- **Current paper draft (UFDM):** [Measuring Statistical Dependencies via Maximum Norm and Characteristic Functions](https://www.researchgate.net/publication/360919080_Measuring_Statistical_Dependencies_via_Maximum_Norm_and_Characteristic_Functions)  
- **Journal publication:** *TODO*

---

### 🛠 Build and Run with Docker

**Build the Docker image and run:**
```bash
docker build -t ufdm .
bash run_docner.sh

### 🛠 Build and Run with Docker

**Build the Docker image and run:**
```bash
docker build -t ufdm .
bash run_docner.sh
```

## 🔬 Experiments from the Paper

### Independence tests (permutation‑based)
`applications/independence_test.py`  
Compares **UFDM** with **DCOR**, **HSIC**, and **MEF**.

### Feature extraction (linear representation learning)
`applications/feature_extraction_dim_selection.py`  
Compares **UFDM**, **DCOR**, **HSIC**, **MEF**, and **Neighborhood Component Analysis (NCA)** for classification tasks.

> **Note:** Datasets for the feature‑extraction experiments are downloaded automatically via the **OpenML API**.

