# 🧠 Bayesian Network Modelling of Risk and Prodromal Markers of Parkinson’s Disease

## 📌 Project Overview

This mini-project implements **Bayesian Network modelling** to study the **risk and prodromal markers of Parkinson’s Disease** using clinical data.
The project is inspired by the research paper *“Bayesian Network Modelling of Risk and Prodromal Markers of Parkinson’s Disease”* and focuses on **probabilistic dependency modelling** and **synthetic data generation**.

A **Streamlit-based web application** is developed to visualize the model and display predictions in an interactive manner.

---

## 🎯 Objectives

* Model dependencies between Parkinson’s risk factors and prodromal markers using a **Bayesian Network**
* Learn probabilistic relationships from real clinical data
* Generate **synthetic data** from the trained Bayesian Network
* Compare real and synthetic data distributions
* Visualize results through a **Streamlit application**

---

## 🧪 Dataset

* Clinical Parkinson’s Disease dataset
* Includes:

  * Risk factors
  * Prodromal markers such as:

    * REM Sleep Behavior Disorder (pRBD)
    * Hyposmia
    * Depression

*(Dataset used only for academic purposes)*

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Framework:** Streamlit
* **Models & Methods:**

  * Bayesian Network
  * Synthetic Data Generation
* **Evaluation Techniques:**

  * Random Forest Classifier
  * AUC / pAUC
  * Kullback–Leibler Divergence (KLD)
* **Libraries:**

  * NumPy
  * Pandas
  * Scikit-learn

---

## 🧠 Methodology

1. Data preprocessing and feature selection
2. Bayesian Network structure learning
3. Parameter learning for probability estimation
4. Synthetic data generation using the Bayesian model
5. Model evaluation using Random Forest classifiers
6. Comparison between real and synthetic data performance
7. Interactive visualization using Streamlit

---

## 📊 Results

* Bayesian Network effectively captured probabilistic relationships between risk factors and prodromal markers
* Synthetic data preserved key statistical properties of real data
* Random Forest classifiers showed comparable AUC values when trained on synthetic data and tested on real data

---

## 🖥️ Streamlit Application

* Interactive interface for:

  * Viewing predictions
  * Exploring prodromal markers
  * Understanding probabilistic outputs
* Enables easy experimentation without command-line interaction

---

## 📄 Reference Paper

**Bayesian Network Modelling of Risk and Prodromal Markers of Parkinson’s Disease**

---

## 📌 Note

This project is developed **solely for academic purposes** as part of the 6th semester mini-project.
