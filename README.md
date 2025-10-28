# Advanced-Financial-Machine-Learning-&-Predictive-Modeling-Pipeline---SPXUSD-(GPU-Accelerated)  
"Developed a GPU-accelerated quantitative research pipeline implementing López de Prado’s Advanced Financial Machine Learning (AFML) framework to forecast SPXUSD market movements through event-based labeling, meta-modeling, and interpretable alpha generation."

---

## 📁 Project Information  
**Self Project — Quantitative Research | 2025**

---

## 🧠 Overview  
This project applies **Marcos López de Prado’s Advanced Financial Machine Learning (AFML)** framework to build a robust, interpretable, and high-frequency predictive modeling system for **SPXUSD**.  

The goal is to forecast **1-minute forward returns** and classify profitable trading events using **GPU-accelerated machine learning models**.  
The system incorporates modern quantitative techniques including **event labeling**, **fractional differentiation**, **Purged K-Fold cross-validation**, and **feature importance diagnostics** for transparency and reliability.  

---

### 1. **Event-Based Labeling Framework**
Implemented **triple-barrier labeling** for event classification to overcome look-ahead bias and ensure realistic label generation.  
Each trade event is terminated by one of three outcomes:  
- **Upper Barrier (Profit Target)**  
- **Lower Barrier (Stop Loss)**  
- **Vertical Barrier (Time Limit)**  

Additionally, **meta-labeling** was applied to enhance the precision of entry signals by classifying the probability of a profitable trade conditioned on the primary model’s output.  

---

### 2. **Feature Engineering**
Generated a diverse set of over **25+ alpha features** capturing short-term market dynamics and volatility, including:  
- Lagged log returns, realized volatility, and autocorrelation  
- Rolling skewness/kurtosis  
- Volume imbalance, bid-ask spread measures  
- Technical indicators (EMA, RSI, Bollinger bands, ATR, etc.)  

Applied **Fractional Differentiation (FFD)** to retain statistical memory while achieving stationarity — validated using **Augmented Dickey-Fuller (ADF) testing**.

---

### 3. **Modeling Pipeline**
Constructed a **GPU-accelerated predictive pipeline** leveraging multiple machine learning algorithms to classify event profitability:  
- **Random Forest (RF)**  
- **XGBoost (XGB)**  
- **LightGBM (LGBM)**  

Integrated:
- **Purged K-Fold Cross Validation with Embargo** to prevent leakage between overlapping events  
- **Sequential Bootstrapping** for realistic sampling of time-series data  
- Hyperparameter optimization for model robustness  

---

### 4. **Feature Importance & Model Explainability**
Performed interpretable alpha diagnostics through multiple importance metrics:  
- **MDI (Mean Decrease Impurity)**  
- **MDA (Mean Decrease Accuracy)**  
- **OFI (Orthogonal Feature Importance)**  

These methods quantify the predictive contribution of each alpha feature, enabling identification of persistent and non-spurious signals driving market returns.

---

### 5. **Performance Evaluation**
- Evaluated classification metrics (Precision, Recall, F1-score) on event outcomes.  
- Analyzed model stability under rolling retraining.  
- Benchmarked against baseline time-series models (Logistic Regression, Ridge).  
- Focused on interpretability and feature persistence across different volatility regimes.  

---

### 6. **Key Concepts**
- **Triple-Barrier Labeling:** Event-based labeling scheme capturing stop, profit, and time limits.  
- **Meta-Labeling:** Secondary model improving signal quality of primary classifiers.  
- **Fractional Differentiation (FFD):** Achieves stationarity without losing market memory.  
- **Purged K-Fold CV:** Prevents label leakage in overlapping time windows.  
- **Sequential Bootstrapping:** Sampling technique preserving temporal dependencies.  

---

### 7. **Tech Stack**
- **Language:** Python  
- **GPU Frameworks:** RAPIDS.ai, CuML, CuDF  
- **Libraries:** `numpy`, `pandas`, `statsmodels`, `xgboost`, `lightgbm`, `sklearn`, `matplotlib`, `scipy`  
- **Environment:** Jupyter Notebook (GPU-enabled)  

---

### 8. **Applications**
- **Algorithmic Trading:** Event-based predictive modeling for alpha generation.  
- **Portfolio Management:** Regime-aware signal filtering for volatility-adjusted strategies.  
- **Quantitative Research:** Application of AFML concepts to real-time data for market prediction.  

---

### 9. **Future Work**
- Integrate **Deep Learning models** (Temporal CNNs, LSTMs) for sequential dependency modeling.  
- Add **Bayesian Hyperparameter Optimization** for improved model selection.  
- Extend framework to **multi-asset pairs** and **regime-adaptive trading systems**.  

---
 
