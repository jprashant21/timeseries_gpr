# Timeseries Forecasting with Gaussian process regression

### Objective: Demand Forecasting

**Python packages:**
  Python-dateutil  
  Pytest  
  Isoweek  
  Sklearn  
  Pandas  
  Numpy  
  Datetime  

Steps:
1. Data processing
2. Seasonality, Trend & Residue extraction
3. Residue modelling
4. Residue prediction
5. Unit test and integration test results


Scripts information:

 - timeseries_modeling_analytics.py  
    Arguments:   
    sample.csv: sample data csv file  
    n: required forecast period for step-2 and step-4  
    Trigger command: python portcast_analytics.py sample.csv 6  

 - test_cases.py  
    Arguments: None  
    Trigger command: pytest test_portcast.py
