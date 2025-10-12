# Question 3: Working with Autoregressive Modeling

## Observations on Autoregressive Temperature Forecasting

An autoregressive (AR) model was implemented using linear regression to forecast the daily minimum temperature in Australia. The core idea was to predict the temperature for day `T+1` based on the temperatures of a preceding window of days (e.g., `T`, `T-1`, `T-2`, ...).

### Model Performance

* **RMSE (Root Mean Squared Error):** The model achieved a reasonable RMSE. This value represents the standard deviation of the prediction errors, giving a quantitative measure of the model's accuracy in degrees Celsius. A lower RMSE indicates a better fit.

* **Predictions vs. True Values Plot:**
    * The plot of predicted temperatures against the true temperatures showed a **strong positive linear relationship**, with most points clustering tightly around the 45-degree line (`y=x`). This indicates that the model's predictions are, on average, closely aligned with the actual outcomes.
    * The model was effective at capturing the general trend and seasonality of the temperature data. When the actual temperature was high, the model predicted a high temperature, and vice-versa.
    * However, the model seemed to have some limitations. It was less accurate at predicting **sudden, sharp changes** or extreme temperature fluctuations. The predictions for these points often appeared slightly "damped" or lagged, meaning the model predicted a value closer to the recent average rather than the true extreme value. This is a characteristic limitation of simple linear AR models, which assume a linear relationship with past values and may not capture complex, non-linear dynamics.



**Conclusion:** Using lagged temperature values as features in a linear regression model provides a solid baseline for time series forecasting. It effectively learns the underlying autocorrelation in the data. For more accurate predictions, especially of extreme events, more advanced models incorporating non-linear relationships or other external features (like humidity, pressure, etc.) would likely be necessary.
