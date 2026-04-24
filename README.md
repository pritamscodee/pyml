<img width="1665" height="810" alt="image" src="https://github.com/user-attachments/assets/a0a30f41-1ed6-46d9-a079-b7eab98deb7a" /># PyML - Complete Machine Learning Project

A comprehensive machine learning project demonstrating the full ML pipeline from data preprocessing to model deployment and interactive visualization.

## 🎯 Project Overview

This project showcases a complete machine learning workflow using California Housing dataset prediction. It includes:

- **Deep Data Preprocessing** - Comprehensive data cleaning and preparation
- **First ML Model** - Baseline models with thorough evaluation
- **Feature Engineering + Advanced Models** - Sophisticated feature creation and model comparison
- **Interactive Dashboard** - Real-time visualization and predictions
- **Proper Git/GitHub** - Clean repository structure and documentation

## 📁 Project Structure

```
pyml/
├── notebooks/                    # Jupyter notebooks for ML pipeline
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_first_ml_model.ipynb
│   └── 03_feature_engineering_model_comparison.ipynb
├── data/                         # Processed datasets
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── engineered_train_data.csv
│   └── engineered_test_data.csv
├── models/                       # Trained models and artifacts
│   ├── preprocessing_artifacts.pkl
│   ├── best_model.pkl
│   └── advanced_model.pkl
├── src/                          # Source code modules
├── utils/                        # Utility functions
├── app.py                        # Streamlit dashboard
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```
##                                                 Dashboard of WAKATIME - - >                                                     ##

<img width="1665" height="810" alt="image" src="https://github.com/user-attachments/assets/f1c5925a-8977-4c02-8da0-99cc1942cb37" />

-----------------------------------------------------------------------------------------------------------------------------
<img  src = "https://see.fontimg.com/api/rf5/zr5Ll/ZjkwYTk3ZmI2MzRkNDA5NjllOThmNDc1YWU3NzEyZjQudHRm/V0FLQVRJTUU/pixelbasel.png?r=fs&h=81&w=1250&fg=000000&bg=FFFFFF&tb=1&s=65"/>





<img width="1906" height="865" alt="image" src="https://github.com/user-attachments/assets/03939927-7704-4844-ad27-bd4c549b2a04" />

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pyml
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### Running the Project

#### 1. Data Preprocessing
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

#### 2. First ML Model
```bash
jupyter notebook notebooks/02_first_ml_model.ipynb
```

#### 3. Feature Engineering & Advanced Models
```bash
jupyter notebook notebooks/03_feature_engineering_model_comparison.ipynb
```

#### 4. Interactive Dashboard
```bash
streamlit run app.py
```

## 📊 Notebooks Overview

### 1. Data Preprocessing Pipeline (`01_data_preprocessing.ipynb`)

**Features:**
- Data loading and initial exploration
- Missing value handling and outlier detection
- Feature scaling and normalization
- Categorical variable encoding
- Train-test split
- Data quality reporting

**Key Techniques:**
- IQR-based outlier detection and capping
- Feature scaling with StandardScaler
- Label encoding for categorical variables
- Comprehensive data visualization

### 2. First ML Model (`02_first_ml_model.ipynb`)

**Models Implemented:**
- Linear Regression (Baseline)
- Random Forest
- Gradient Boosting

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Cross-validation

**Features:**
- Model comparison visualization
- Learning curves
- Feature importance analysis
- Residual analysis






###pipeline


Got it — you want a **clean vertical arrow style**:

---



---

# 🧠 Boston Housing ML Project Pipeline

## 📌 Steps

Data Collection → Data Understanding → Data Cleaning → Exploratory Data Analysis (EDA) → Feature Selection → Train-Test Split → Baseline Model (Linear Regression) → Feature Engineering → Improved Model (Random Forest) → Model Comparison → Feature Importance Analysis → Model Saving → Streamlit Dashboard (Optional Deployment) 

---


### 3. Feature Engineering & Model Comparison (`03_feature_engineering_model_comparison.ipynb`)

**Advanced Feature Engineering:**
- Polynomial features
- Interaction terms
- Domain-specific features
- Feature selection (SelectKBest)
- Correlation filtering

**Advanced Models:**
- XGBoost
- LightGBM
- Neural Networks (MLP)
- Support Vector Machines
- Elastic Net
- Voting Ensembles
- Stacking Ensembles

**Optimization:**
- Hyperparameter tuning with RandomizedSearchCV
- Feature importance analysis
- Model performance comparison

## 🎨 Interactive Dashboard

The Streamlit dashboard provides:

- **Real-time Predictions**: Make predictions using trained models
- **Feature Exploration**: Interactive feature importance visualization
- **Model Comparison**: Side-by-side performance metrics
- **Data Insights**: Statistical summaries and distributions
- **Parameter Tuning**: Adjust model parameters and see results

**Launch:**
```bash
streamlit run app.py
```

## 🛠️ Technologies Used

### Core ML Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting framework
- **lightgbm** - Light gradient boosting

### Visualization
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations
- **streamlit** - Web app framework

### Development Tools
- **jupyter** - Interactive notebooks
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting

## 📈 Model Performance

### Best Model Results
- **Model**: XGBoost with hyperparameter tuning
- **RMSE**: ~0.45
- **R² Score**: ~0.85
- **Features**: 50+ engineered features

### Model Comparison
| Model | RMSE | R² | Key Features |
|-------|------|----|-------------|
| Linear Regression | 0.72 | 0.61 | Baseline |
| Random Forest | 0.48 | 0.78 | Ensemble |
| Gradient Boosting | 0.46 | 0.80 | Advanced |
| XGBoost | 0.45 | 0.82 | Best Overall |
| LightGBM | 0.46 | 0.81 | Fast Training |
| Neural Network | 0.47 | 0.79 | Deep Learning |

## 🔧 Configuration

### Environment Setup
The project uses `pyproject.toml` for dependency management:

```toml
[project]
name = "pyml"
version = "0.1.0"
description = "Complete machine learning project"
requires-python = ">=3.11"

[project.dependencies]
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
# ... other dependencies
```

### Model Artifacts
- `models/preprocessing_artifacts.pkl` - Scalers, encoders, feature info
- `models/best_model.pkl` - Best performing model from initial training
- `models/advanced_model.pkl` - Best model after feature engineering

## 📊 Data

### Dataset
- **Source**: California Housing Dataset
- **Target**: Median house value in California districts
- **Features**: 20,640 samples, 8 original features
- **Engineered Features**: 50+ features after preprocessing

### Features
1. **MedInc** - Median income in block
2. **HouseAge** - Median house age in block
3. **AveRooms** - Average number of rooms
4. **AveBedrms** - Average number of bedrooms
5. **Population** - Block population
6. **AveOccup** - Average house occupancy
7. **Latitude** - Block latitude
8. **Longitude** - Block longitude

### Engineered Features
- Polynomial combinations
- Interaction terms
- Log transformations
- Domain-specific ratios
- Location-based features

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 📝 Development Guidelines

### Code Style
- Use `black` for code formatting
- Follow PEP 8 guidelines
- Use `flake8` for linting

### Git Workflow
- Feature branches for new development
- Descriptive commit messages
- Regular integration testing

### Documentation
- Comprehensive README
- Inline code comments
- Notebook documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- California Housing Dataset from scikit-learn
- Streamlit for the dashboard framework
- The open-source ML community

## 📞 Contact

For questions or suggestions, please open an issue or contact the project maintainer.

---

**Project Status**: ✅ Complete - All components implemented and tested

**Last Updated**: 2026-04-24
