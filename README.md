# Delhi House Price Prediction Using Machine Learning

An interactive Streamlit application to estimate residential property prices in Delhi based on historical MagicBricks data.

## Project Architecture
- `app.py`: Streamlit User Interface
- `house_price_pipeline.pkl`: Trained Random Forest Regressor Pipeline
- `category_options.json`: Pre-extracted categorical features for UI consistency
- `requirements.txt`: Python package dependencies with verified versions
- `model_comparison.csv`: Performance metrics comparing Linear Regression and Random Forest

## Performance Summary
- **Random Forest R²**: 0.8337
- **Random Forest MAE**: ₹6,324,771.74
- **Random Forest RMSE**: ₹12,383,640.96

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run application: `streamlit run app.py`
