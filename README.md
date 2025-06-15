# Corporación Favorita Grocery Sales Forecasting
This project focuses on building time series models to forecast item-level sales for Corporación Favorita grocery stores across Ecuador. Accurate sales forecasting helps optimize inventory management, prevent stockouts, and enhance promotional strategies.

## Objective
The primary goal is to predict future sales of items sold in different store locations using historical sales data. Reliable forecasts support better decision-making for stock, logistics, and business planning.

## 📊 Dataset
Source: Kaggle Competition - [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

Download Method: Via Kaggle API script (kaggle competitions download)

The dataset includes:
| Dataset | Description    |
| :-------- | :------- |
| train.csv:| Daily item-level sales (unit_sales) per store. Includes onpromotion. Only non-zero sales are recorded. Negative values indicate returns.|
| stores.csv:| city, state, type, and cluster.|
| items.csv: | Item metadata – family, class, and perishable. Perishable items have a score weight of 1.25; others, 1.0.|
| transactions.csv:| Number of transactions per store/date (training period only).|
| oil.csv: | Daily oil prices. Important due to Ecuador's oil-driven economy.|
| holidays_events.csv: | National/local holidays and special events, including transferred, bridge, workday, and additional holiday types.|

### Downloading Additional File
 This script requires an additional file state_dict_model.pt* to execute.
 Download the file from this [link]([https://drive.google.com/file/d/1-OZfY3-VOYt44nThkkuhO5z_QbXs1e4e/view?usp=sharing)] and place the downloaded file into a folder named Data in the root directory of the project.
 * Afterwards your directory should look something like this:
```bash
Data/
├── Data/
│   └── your_downloaded_data_file.csv
├── Time_Series_Project_Data_Analyse_EDA(2).ipynb
├── Time_Series_Project_Data_Preprocessing(3).ipynb
...
├── ..............
├── ..............
```

## Project Structure

Project Structure
This project is broken down into several stages for clarity and modular analysis:

| Notebook | Description    |
| :-------- | :------- |
| Time_Series_project_Kaggle_API_data_download.ipynb | Kaggle API to download the dataset for project, including steps to authenticate, fetch, and extract the data|
| Time_Series_Project_Filter_Traindata(1).ipynb | Focused on filtering train data for Guayas region and top 3 item families|
| Time_Series_Project_Data_Analyse_EDA(2).ipynb | Exploratory Data Analysis (EDA): trends, seasonality, outliers, and missing values|
| Time_Series_Project_Data_Preprocessing(3).ipynb | Handling missing values, handling outliers, formatting dates, cleaning and basic preprocessing steps|
| Time_Series_Project_Feature_Engineering(4).ipynb | Creation of time-based features, lags, rolling stats, exponential smoothing, holiday indicators, etc|
| Time_Series_Project_Sarimax_Holtwinters_model(5).ipynb | Time series forecasting using SARIMAX and Holt-Winters models|
| Time_Series_Project_XGboost_Model(6).ipynb | Gradient boosting model (XGBoost) for forecasting with hyperpamrameter tunning|


## 🔧 Key Techniques Used
* Time-based train-test split

* Log transformation for stabilizing variance

* Feature scaling (StandardScaler)

* Lag and rolling window feature engineering

* Exogenous variable selection via correlation analysis


## Forecasting Models
SARIMAX: Seasonal AutoRegressive Integrated Moving Average with exogenous variables

Holt-Winters: Triple exponential smoothing

XGBoost: Machine learning model with custom time features

## 📈  Evaluation Metrics
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* R² Score

## Getting Started

### Running the Script
To run the Python script follow these steps:
### Step 1: Clone the GitHub Repository 
* Clone the repository to your local machine using Git.
* Open command prompt and run the following command:
```bash
git clone https://github.com/NishatFarihaRimi/time_series_forecasting_1.git
```
### Step 2: Navigate to the Cloned Directory 
```bash
cd time_series_forecasting_1
```
Clone this repo:

pip install -r requirements.txt
Run notebooks in order for full pipeline (EDA → Preprocessing → Feature Engineering → Modeling)

### Step 3: Set Up a Virtual Environment (Optional)
* Creating a venv with conda and activating it
```bash
conda create -n time_series_env python==3.8.0
conda activate time_series_env
```
### Step 4: Install dependencies
 Installing the packages listed in *'requirements.txt'** file
```bash
pip install -r requirements.txt
```

## 📌 Requirements
* Python 3.8+
* pandas, numpy, matplotlib, seaborn
* statsmodels, scikit-learn
* xgboost
* kaggle (for data download)

## ✨ Future Work
* Hyperparameter tuning using GridSearchCV
* Add LSTM/DeepAR models
* Deploy model via Streamlit or Flask

## Acknowledgments
Dataset from Kaggle: Corporación Favorita Grocery Sales Forecasting
