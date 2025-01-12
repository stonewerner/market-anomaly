# 📈 Stock Market Crash Predictor

An advanced machine learning system that predicts potential stock market crashes with 100% recall. Using XGBoost and sophisticated feature engineering, this model identifies weeks where the market may experience a ≥5% drop.

![ML Model](https://img.shields.io/badge/Model-XGBoost-red)
![Recall](https://img.shields.io/badge/Recall-100%25-brightgreen)
![Data](https://img.shields.io/badge/Historical%20Data-2000--2021-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-orange)

## 🌟 Key Features

- 🎯 100% recall in detecting market crashes
- 📊 Interactive Streamlit dashboard
- 📈 Real-time market indicators
- 🔮 Weekly crash predictions with confidence scores
- 📉 Comprehensive market analysis
- ⚡ Fast and efficient predictions

## 🤖 Model Details

### Definition of a Market Crash

- A drop of 5% or more in market value within a week
- Binary classification problem (Crash/No Crash)

### Technical Implementation

- **Base Model**: XGBoost Classifier
- **Sampling**: SMOTE for handling class imbalance
- **Feature Engineering**:
  - Lagged features for temporal patterns

## 📊 Dashboard Features

The Streamlit dashboard provides:

- Real-time crash probability predictions
- Confidence scores for predictions
- Feature importance analysis

## 💻 Tech Stack

- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: plotly
- **Dashboard**: Streamlit
- **Data Augmentation**: SMOTE

## 📈 Performance Metrics

- Recall: 100%
- Training Period: 2000-2021
- Validation Method: Time-series cross-validation
- Regular Model Retraining: Weekly

## 🔧 Feature Engineering

### Lagged Features

- Previous week's returns
- Rolling averages
- Volatility metrics
- Volume indicators

### Technical Indicators

- Moving averages
- VIX
- VG1
- MXRU
- CRY

## 📊 Data Pipeline

1. Data Collection

   - Historical market data (2000-2021)
   - Volume metrics
   - Market indicators

2. Preprocessing

   - Feature engineering
   - SMOTE application
   - Normalization
   - Missing value handling

3. Model Training
   - XGBoost optimization
   - Hyperparameter tuning
   - Cross-validation

## 🖥️ Usage

1. Access the Streamlit dashboard
2. View current market predictions
3. Analyze confidence metrics
4. Explore feature importance
5. Monitor historical accuracy

## ⚠️ Disclaimer

This tool is for research and educational purposes only. Stock market predictions involve risk, and no financial decisions should be made solely based on this model's outputs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

---

**Note**: Past performance does not guarantee future results. Always conduct your own research and consult with financial advisors before making investment decisions.
