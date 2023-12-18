## Univariate-Time-Series-Forecasting
# Los Angeles Crime Forecasting
Crime forecasting plays a pivotal role in enhancing policing decision-making processes. The primary objective of this project is to develop a two-week crime forecasting model. Our aim is to predict the incidents of crime that may occur in Los Angeles.

To achieve this, we leverage a publicly available dataset titled "Crime Data from 2020 to Present," accessible via the website: https://data.lacity.org/. This dataset provides a comprehensive record of reported crimes within the City of Los Angeles, dating back to 2020.

It's worth noting that this data is transcribed from original crime reports, which are traditionally documented on paper. Consequently, there may be some inherent inaccuracies in the dataset. However, these inaccuracies are unlikely to impact the performance of our approach significantly, as they primarily pertain to unused variables.

The univariate dataset we work with comprises two main components:

"Occurrences": This variable represents the daily total number of crimes reported in Los Angeles.

Date Index: A chronological reference for tracking the timing of these crime occurrences.

By focusing on these key elements, we aim to develop a robust crime forecasting model that can assist law enforcement agencies in proactive decision-making and resource allocation.

The model we propose is detailed in the file titled "Los Angeles Crime Forecasting" above. Our approach begins with data pre-processing, which includes managing outliers, addressing missing values, and transforming the time series into a stationary form.

Subsequently, we apply five widely recognized algorithms for univariate time series analysis. The outcomes of these algorithms are then visualized and presented in the final stage of our deployment process.

![Alt Text](https://github.com/KovousoglouGeorgios/Univariate-Time-Series-Forecasting/blob/3537c1c68ed4d7208606e504dcc7d3b68109f9b3/results%20plot.jpg)

In summary, univariate time series forecasting presents a complex challenge that requires careful consideration. It is essential to emphasize the pivotal role of data preprocessing in this analysis. Statistical models typically operate under the assumption of stationarity in the time series. In our study, we achieved stationarity through differencing the dataset.

 ![Alt Text](https://github.com/KovousoglouGeorgios/Univariate-Time-Series-Forecasting/blob/d778482f309db66d2caea88fed4a81c5c460ead5/results%20table.jpg)

Regarding model selection, it is our belief that the various models demonstrated comparable performance. While the AutoARIMA model emerged as the most efficient option, its performance did not exhibit a significant divergence from the other models.
