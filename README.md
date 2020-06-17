# Load Forecasting During COVID-19
This is the code repo for [Using Mobility for Electrical Load Forecasting During the COVID-19 Pandemic](https://arxiv.org/abs/2006.08826)

By Yize Chen, Weiwei Yang and Baosen Zhang, 

University of Washington and Microsoft Research

The method shown in this repository can be used for improving load forecasting performance during the uncertain times due to COVID-19 pandemic.

### Introduction
The COVID-19 pandemic has brought major disruptions to the energy systems' operations. To keep the lights on during the difficult times, power system engineers rely on accurate load forecasts to schedule electricity generations. However, the change of mobility patterns, the implementations of stay-at-home orders have greatly reshaped the electricity consumption, and made an accurate forecast of future loads challenging. For example, based on the public data provided by California Independent System Operator (CAISO), it overforecasted the system load consistenly in April 2020.

<img src="./images/forecast_error_2.png" width="500">

We propose a novel load forecasting model by integrating the mobility features to inform the forecast model of socioeconomic changes. We also use multi-task learning to enable knowledge transfer between different load regions. 


### Data
There are several data sources listed in 'Data' folder. We included 12 regions of different size and load patterns for evaluation. The load data is coming from utilities and ISOs, while the weather forecast data is coming from WorldWeatherOnline historical weather data API.


### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* TensorFlow v1.0+
* Keras
* wwo_hist


### Run Experiments with Pre-Processed Datasets
Construct the dataset using features from mobility
```
python scripts/Construct_dataset.py
```

Train the multi-task learning model based on mobility data
```
python scripts/transfer_main.py
```

Compare and plot the results of different algorithms
```
python scripts/plot_forecast.py
```

<img src="./images/Forecast_result.png" width="500">


## Questions?

Please email [me](http://blogs.uw.edu/yizechen/) at yizechen@uw.edu if you have any code or implemenration questions!
