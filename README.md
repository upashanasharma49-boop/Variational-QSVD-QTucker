# Variational-QSVD-QTucker

## Overview

This repository contains implementations of Variational Quantum Singular Value Decomposition (QSVD) and Quantum Tucker Decomposition (QTucker) for matrix and tensor factorization using parameterized quantum circuits.

The work investigates variational quantum approaches for extracting Schmidt spectra, computing entanglement measures, and performing tensor decompositions inspired by the classical Singular Value Decomposition (SVD) and Higher-Order Singular Value Decomposition (HOSVD).

---

## Features

### Variational Quantum Singular Value Decomposition (QSVD)

- Schmidt spectrum extraction of bipartite quantum states
- Variational optimization using parameterized quantum circuits
- Computation of Von Neumann entanglement entropy
- Comparison against classical SVD benchmarks
- Validation on:
  - 8-qubit bipartite states (16×16 Schmidt matrices)
  - 10-qubit bipartite states corresponding to 32×32 Schmidt matrices

### Quantum Tucker Decomposition (QTucker)

- Variational quantum analogue of Tucker decomposition
- Mode-wise tensor factorization using QSVD
- Construction of factor matrices and core tensors
- Comparison with classical HOSVD results

---

## Requirements

- Python 3.10+
- NumPy
- SciPy

Install dependencies using

```bash
pip install numpy scipy
```

---

## Running QSVD

Execute

```bash
python qsvd.py
```

The program supports

- Random quantum states
- User-defined matrices

and outputs

- Schmidt coefficients
- Von Neumann entropy
- Classical SVD comparison
- Relative errors

---

## Numerical Validation

The QSVD implementation was tested on randomly generated bipartite quantum states and reproduces singular values and entanglement entropies obtained from classical SVD calculations with high numerical accuracy.

---

## Repository Structure

qsvd.py
qtucker.py
examples/
README.md

---

## Disclaimer

The variational quantum circuits are simulated classically using NumPy and SciPy. The implementation is intended as a proof-of-concept study of variational quantum matrix and tensor decomposition methods.

---

