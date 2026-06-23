# Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

**Domain:** E-Commerce and Retail Analytics  
**Type:** Unsupervised Machine Learning  
**Contributor:** Anveiksha Sharma  

---

## Project Overview

Shopper Spectrum is an end-to-end data science project that analyzes real-world online retail transaction data to uncover customer purchasing behavior, segment customers into meaningful groups, and recommend similar products — all deployed through an interactive Streamlit web application.

---

## Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily. Most businesses struggle to extract meaningful intelligence from this data, leading to poor customer retention, wasted marketing budgets, and low product discovery rates.

This project solves two core problems:
1. **Customer Segmentation** — Group customers based on their purchasing behavior using RFM analysis and KMeans clustering
2. **Product Recommendation** — Suggest top 5 similar products for any given product using collaborative filtering

---

## Dataset

The dataset contains **541,909 online retail transactions** from December 2022 to December 2023 across 38 countries.

| Column | Description |
|---|---|
| InvoiceNo | Transaction number |
| StockCode | Unique product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit (£) |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

> **Note:** Dataset was provided by Labmentix instructor. To run locally, place `online_retail.csv` in the root folder.

---

## Project Structure

```
Shopper_Spectrum/
├── Shopper_Spectrum_EDA.ipynb       # Exploratory Data Analysis notebook
├── Shopper_Spectrum_ML.ipynb        # Machine Learning notebook
├── app.py                           # Streamlit web application
├── kmeans_model.pkl                 # Trained KMeans clustering model
├── scaler.pkl                       # Fitted StandardScaler
├── cluster_labels_map.pkl           # Cluster to segment label mapping
├── product_similarity.pkl           # Product cosine similarity matrix
└── README.md
```

---

## Methodology

### Part 1 — Customer Segmentation (RFM + KMeans)

1. **Data Cleaning** — Removed missing CustomerIDs, cancelled invoices, and invalid quantities/prices
2. **RFM Feature Engineering:**
   - **Recency** = Days since customer's last purchase
   - **Frequency** = Number of distinct transactions
   - **Monetary** = Total amount spent
3. **Standardization** — Applied StandardScaler to normalize RFM features
4. **Clustering** — Used Elbow Method + Silhouette Score to select k=4
5. **Labeling** — Clusters labeled based on RFM averages:

| Segment | Characteristics |
|---|---|
| High-Value | Recent, frequent, high spenders |
| Regular | Steady purchasers, moderate spend |
| Occasional | Rare, low-value purchases |
| At-Risk | Haven't purchased in a long time |

**Final Silhouette Score: 0.616**

---

### Part 2 — Product Recommendation (Collaborative Filtering)

1. Built a **CustomerID × Product** purchase matrix
2. Computed **cosine similarity** between all products
3. Given any product name → returns **top 5 most similar products**

---

## Streamlit App

The app has two interactive modules:

### Product Recommendation Module
- Enter a product name
- Click **Get Recommendations**
- Returns 5 similar products with similarity scores

### Customer Segmentation Module
- Enter Recency, Frequency, Monetary values
- Click **Predict Cluster**
- Returns customer segment label with business description

---

## How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/anveiksha-max/Shopper-Spectrum.git
cd Shopper-Spectrum
```

**2. Install dependencies:**
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
```

**3. Place `online_retail.csv` in the root folder**

**4. Run the app:**
```bash
streamlit run app.py
```

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | KMeans clustering, StandardScaler, cosine similarity |
| Streamlit | Web application deployment |
| Pickle | Model serialization |
| Google Colab | Development environment |

---

## Key Insights

- UK accounts for the majority of transactions (~90%)
- Top 10 products drive a disproportionate share of total sales (80/20 pattern)
- Sales peak in October–November (holiday season)
- Most customers are low-frequency, low-spend buyers — only 13 customers qualify as High-Value
- Cosine similarity scores above 0.6 for top recommendations confirm strong product relevance

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| KMeans Clustering (k=4) | Silhouette Score | 0.616 |
| KMeans Clustering (k=4) | Inertia | 4092.14 |
| Collaborative Filtering | Avg Similarity Score (top 5) | 0.60 – 0.85 |

---

*Submitted as part of Labmentix AI/ML Internship Capstone Project*
