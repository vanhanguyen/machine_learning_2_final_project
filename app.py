# streamlit_app.py
import io
import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import seaborn as sns
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#st.write(matplotlib.__version__)

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("Time Series Forecasting App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your time series CSV (with datetime index)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data Preview:", df.head())

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif df.columns[0].lower().startswith("date"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df.set_index(df.columns[0], inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in the uploaded file. Please upload a time series with numeric values.")
        st.stop()

    st.write("Adjusted Closing Price Over Time")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['adjClose'], label='Adjusted Closing', color='#FF914D')
    ax.set_title('Yahoo Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)

    # Format the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15, interval=3))
    plt.xticks(rotation=45)

    st.pyplot(fig)


    #Trend identification
    st.write("Adjusted Closing Price with Trend Lines")

    # Calculate moving averages
    rolling_mean_12 = df['adjClose'].rolling(window=12).mean()
    rolling_mean_60 = df['adjClose'].rolling(window=60).mean()

    # Plot
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(df['adjClose'], label='Original Data', color='orange')
    ax2.plot(rolling_mean_12, label='Trend (12-period MA)', color='green', linestyle='-.')
    ax2.plot(rolling_mean_60, label='Trend (60-period MA)', color='blue', linestyle='-.')

    ax2.set_title('Adjusted Closing Price with Trend Line', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Adjusted Closing Price', fontsize=12)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    st.pyplot(fig2)

    #Stationarity
    #Augmented Dickey-Fuller (ADF) test:
    st.write("Augmented Dickey-Fuller (ADF) Test for Stationarity")

    # Run ADF test
    adf_result = adfuller(df['adjClose'].dropna())

    # Display results
    st.write(f"**ADF Statistic:** {round(adf_result[0], 3)}")
    st.write(f"**p-value:** {round(adf_result[1], 3)}")

    st.write("**Critical Values:**")
    for key, value in adf_result[4].items():
        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{key}: {round(value, 3)}")

    # Optional stationarity interpretation
    if adf_result[1] < 0.05:
        st.success("The series is likely stationary (reject null hypothesis).")
    else:
        st.warning("The series is likely non-stationary (fail to reject null hypothesis).")




    #Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    st.write("KPSS Test for Trend Stationarity")

    # Run KPSS test
    kpss_result = kpss(df['adjClose'].dropna(), regression='ct')  # 'ct' = constant + trend

    # Display results
    st.write(f"**KPSS Statistic:** {round(kpss_result[0], 2)}")
    st.write(f"**p-value:** {kpss_result[1]}")

    st.write("**Critical Values:**")
    for key, value in kpss_result[3].items():
        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{key}: {value}")

    # Optional interpretation
    if kpss_result[1] < 0.05:
        st.warning("The series is likely **not** trend-stationary (reject null hypothesis).")
    else:
        st.success("The series is likely trend-stationary (fail to reject null hypothesis).")


    #Time Series Decomposition
    st.write("Kolmogorov–Smirnov (K-S) Test for Strict Stationarity")


    # Function to split the series and run the K-S test
    def ks_test_stationarity(series):
        split = len(series) // 2
        series_first_half = series[:split]
        series_second_half = series[split:]
        stat, p_value = ks_2samp(series_first_half, series_second_half)
        return stat, p_value


    # Run on original series
    ks_stat_strict, ks_pvalue_strict = ks_test_stationarity(df['adjClose'])

    # Create a simulated non-strictly stationary series
    non_strict_values = np.concatenate([
        df['adjClose'].iloc[:len(df) // 2].values,
        df['adjClose'].iloc[len(df) // 2:].values * 1.5
    ])
    non_strict_stationary_series = pd.Series(non_strict_values, index=df.index)

    # Run test on simulated non-stationary version
    ks_stat_non_strict, ks_pvalue_non_strict = ks_test_stationarity(non_strict_stationary_series)

    # Plot both series
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(df['adjClose'], color='#FF914D')
    axes[0].set_title(f"Strict Stationary Series (adjClose) - K-S p-value: {ks_pvalue_strict:.4f}")

    axes[1].plot(non_strict_stationary_series, color='#FF914D')
    axes[1].set_title(f"Non-Strict Stationary Series (Simulated) - K-S p-value: {ks_pvalue_non_strict:.4f}")

    plt.tight_layout()
    st.pyplot(fig)

    # Display statistics
    st.write("**Strict Stationary Series (Original)**")
    st.write(f"K-S Test Statistic: `{ks_stat_strict:.4f}`, p-value: `{ks_pvalue_strict:.4f}`")

    st.write("**Non-Strict Stationary Series (Simulated)**")
    st.write(f"K-S Test Statistic: `{ks_stat_non_strict:.4f}`, p-value: `{ks_pvalue_non_strict:.4f}`")

    # Interpretation
    if ks_pvalue_strict < 0.05:
        st.warning("The original series may **not** be strictly stationary (K-S p < 0.05).")
    else:
        st.success("The original series appears strictly stationary (K-S p >= 0.05).")


    # Decomposition
    model_type = st.radio("Choose decomposition model:", ["Additive", "Multiplicative"])

    try:
        ts = df["adjClose"].dropna()  # Always use adjClose
        decomposition = seasonal_decompose(ts, model=model_type.lower(), period=30)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        axs[0].plot(ts, color='#FF914D')
        axs[0].set_title("Original")
        axs[1].plot(trend, color='#FF914D')
        axs[1].set_title("Trend")
        axs[2].plot(seasonal, color='#FF914D')
        axs[2].set_title("Seasonality")
        axs[3].plot(residual, color='#FF914D')
        axs[3].set_title("Residual")

        for ax in axs:
            ax.legend([ax.get_title()])
            ax.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Decomposition failed: {e}")


    #Seasonal and Trend decomposition using Loess (STL) Time Series Decomposition
    from statsmodels.tsa.seasonal import STL

    st.write("STL Decomposition (Seasonal-Trend Decomposition using Loess)")

    try:
        ts = df["adjClose"].dropna()
        stl = STL(ts, period=30)
        result = stl.fit()

        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        axs[0].plot(result.observed, color='#FF914D')
        axs[0].set_title("Original")
        axs[1].plot(result.trend, color='#FF914D')
        axs[1].set_title("Trend")
        axs[2].plot(result.seasonal, color='#FF914D')
        axs[2].set_title("Seasonality")
        axs[3].plot(result.resid, color='#FF914D')
        axs[3].set_title("Residual")

        for ax in axs:
            ax.legend([ax.get_title()])
            ax.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"STL Decomposition failed: {e}")




    #Modeling
    #Differencing
    st.write("Differencing (Lag-1)")

    # Perform differencing to make the series stationary
    df['adjClose_diff'] = df['adjClose'].diff()

    # Display the first 10 rows to inspect
    st.write("**Original vs Differenced Series (First 10 Rows):**")
    st.dataframe(df[['adjClose', 'adjClose_diff']].head(10))

    # Plot the differenced series
    fig_diff, ax_diff = plt.subplots(figsize=(12, 6))
    ax_diff.plot(df.index, df['adjClose_diff'], label='Differenced Series', color='purple')
    ax_diff.set_title('Differenced Series (Lag-1)', fontsize=16)
    ax_diff.set_xlabel('Date')
    ax_diff.set_ylabel('Differenced adjClose')
    ax_diff.legend()
    ax_diff.grid(True)

    st.pyplot(fig_diff)

    #Transformation
    st.write("Data Transformations")

    prices = df['adjClose'].dropna()

    # Apply transformations
    df['adjClose_log'] = np.log(prices)
    df['adjClose_sqrt'] = np.sqrt(prices)

    # Track which columns to show
    transform_cols = ['adjClose', 'adjClose_log', 'adjClose_sqrt']

    # Box-Cox transformation (only for positive prices)
    try:
        df['adjClose_boxcox'], boxcox_lambda = stats.boxcox(prices[prices > 0])
        df['boxcox_lambda'] = boxcox_lambda
        transform_cols.append('adjClose_boxcox')
        st.success(f" Box-Cox transformation applied (lambda = {round(boxcox_lambda, 4)})")
    except Exception as e:
        st.error(f"Box-Cox transformation failed: {e}")

    # Display transformed values
    st.write("**Preview of Transformed Data (First 5 Rows):**")
    st.dataframe(df[transform_cols].head())

    # Plot transformations
    fig_transform, axs = plt.subplots(len(transform_cols) - 1, 1, figsize=(14, 10), sharex=True)

    colors = ['green', 'blue', 'purple']
    titles = transform_cols[1:]

    for i, col in enumerate(titles):
        axs[i].plot(df.index, df[col], color=colors[i % len(colors)])
        axs[i].set_title(f"{col} Transformed")

        axs[i].grid(True)

    plt.tight_layout()
    st.pyplot(fig_transform)


    #De-trending
    st.write("Detrending the Time Series")

    prices = df['adjClose'].dropna()

    # Detrending using linear trend
    try:
        trend = np.polyfit(np.arange(len(prices)), prices, 1)
        trendline = np.polyval(trend, np.arange(len(prices)))
        df['adjClose_detrended_linear'] = prices - trendline
        st.success("Linear trend detrending applied.")
    except Exception as e:
        st.error(f"Linear detrending failed: {e}")

    # Detrending using Moving Average
    window = 30
    try:
        moving_average = prices.rolling(window=window).mean()
        detrended_col = f'adjClose_detrended_MA{window}'
        df[detrended_col] = prices - moving_average
        st.success(f" Moving Average detrending applied (Window = {window}).")
    except Exception as e:
        st.error(f"Moving Average detrending failed: {e}")

    # Display data
    st.write("**Preview of Detrended Data (First 5 Rows):**")
    cols_to_show = ['adjClose', 'adjClose_detrended_linear', detrended_col]
    st.dataframe(df[cols_to_show].dropna().head())

    # Plot detrended results
    fig_det, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(df.index, df['adjClose_detrended_linear'], color='red')
    axs[0].set_title("Linear Detrended Series")

    axs[1].plot(df.index, df[detrended_col], color='darkorange')
    axs[1].set_title(f"Moving Average Detrended Series (Window = {window})")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig_det)


    #Seasonal adjustment
    # --- Seasonal Adjustment using Decomposition ---
    st.write("Seasonal Adjustment using Decomposition")

    prices = df['adjClose'].dropna()
    window = 30  # You can also make this dynamic if you want

    try:
        decomposition = seasonal_decompose(prices, model='additive', period=window)
        adjusted_col = f'adjClose_seas_adjusted{window}'
        df[adjusted_col] = prices / decomposition.seasonal
        df[adjusted_col] = df[adjusted_col].dropna()

        st.success(f" Seasonal adjustment completed using decomposition (Period = {window}).")

        # Display adjusted data
        st.write("**Preview of Seasonally Adjusted Series:**")
        st.dataframe(df[['adjClose', adjusted_col]].dropna().head())

        # Plot original vs adjusted
        fig_seas_adj, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df['adjClose'], label='Original adjClose', color='#FF914D')
        ax.plot(df.index, df[adjusted_col], label=f'Seasonally Adjusted (period={window})', color='green')
        ax.set_title(f'Seasonal Adjustment (Period = {window})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_seas_adj)

    except Exception as e:
        st.error(f"Seasonal decomposition failed: {e}")


    #Random Walk and White Noise
    st.write("Stationarity Tests (ADF, KPSS, Ljung-Box)")

    columns_to_test = [
        'adjClose',
        'adjClose_detrended_linear',
        'adjClose_detrended_MA30',
        'adjClose_seas_adjusted30'
    ]


    # --- ADF Test ---
    def adf_test(series):
        result = adfuller(series)
        return {
            'stat': result[0],
            'pvalue': result[1],
            'crit_values': result[4]
        }


    # --- KPSS Test ---
    def kpss_test(series):
        result = kpss(series, regression='c')
        return {
            'stat': result[0],
            'pvalue': result[1],
            'crit_values': result[3]
        }


    # --- Run tests for each column ---
    for col in columns_to_test:
        series = df[col].dropna()

        st.write(f" Stationarity Tests for `{col}`")

        # ADF
        adf_res = adf_test(series)
        st.write("**ADF Test (Unit Root Test)**")
        st.write(f"- ADF Statistic: `{adf_res['stat']:.4f}`")
        st.write(f"- p-value: `{adf_res['pvalue']:.4f}`")
        st.write("- Critical Values:")
        for k, v in adf_res['crit_values'].items():
            st.write(f"  - {k}: {round(v, 4)}")
        if adf_res['pvalue'] < 0.05:
            st.success("ADF: Series is likely stationary (reject null).")
        else:
            st.warning("ADF: Series may be non-stationary (fail to reject null).")

        # KPSS
        try:
            kpss_res = kpss_test(series)
            st.write("**KPSS Test (Level Stationarity Test)**")
            st.write(f"- KPSS Statistic: `{kpss_res['stat']:.4f}`")
            st.write(f"- p-value: `{kpss_res['pvalue']:.4f}`")
            st.write("- Critical Values:")
            for k, v in kpss_res['crit_values'].items():
                st.write(f"  - {k}: {round(v, 4)}")
            if kpss_res['pvalue'] < 0.05:
                st.warning("KPSS: Series is likely non-stationary (reject null).")
            else:
                st.success("KPSS: Series is likely stationary (fail to reject null).")
        except Exception as e:
            st.error(f"KPSS Test failed for {col}: {e}")

        # Ljung-Box
        st.write("**Ljung-Box Test (Autocorrelation Check)**")
        lb_result = acorr_ljungbox(series, lags=[10], return_df=True)
        st.dataframe(lb_result)

    # --- Simulating White Noise and Random Walk ---
    st.write("Simulated Series: White Noise vs Random Walk")

    np.random.seed(0)
    n = 1000

    white_noise = np.random.normal(0, 1, n)
    random_shocks = np.random.normal(0, 1, n)
    random_walk = np.cumsum(random_shocks)

    # Plot both
    fig_sim, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(white_noise, color='#FF914D')
    ax1.set_title("White Noise")
    ax2.plot(random_walk, color='#FF914D')
    ax2.set_title("Random Walk")
    st.pyplot(fig_sim)


    # ADF & KPSS & Ljung-Box on Simulated Data
    def display_sim_test(name, series):
        st.write(f" Tests on {name}")
        adf_res = adf_test(series)
        kpss_res = kpss_test(series)
        lb = acorr_ljungbox(series, lags=[10], return_df=True)

        st.write(f"- **ADF p-value**: {adf_res['pvalue']:.4f}")
        st.write(f"- **KPSS p-value**: {kpss_res['pvalue']:.4f}")
        st.write("**Ljung-Box Result (lag 10):**")
        st.dataframe(lb)


    display_sim_test("White Noise", white_noise)
    display_sim_test("Random Walk", random_walk)

    st.subheader("Ljung-Box Test for Autocorrelation")

    # Perform Ljung-Box test for 'adjClose' column with lag 10
    try:
        lb_result = acorr_ljungbox(df['adjClose'].dropna(), lags=[10], return_df=True)
        st.write("**Ljung-Box Test Result (Lag = 10):**")
        st.dataframe(lb_result)
    except Exception as e:
        st.error(f"Ljung-Box test failed: {e}")

    #ACF and PACF Plot
    st.subheader("ACF and PACF Plots")

    # Compute the differenced series
    series = df['adjClose_seas_adjusted30'].diff().diff().dropna()

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ACF Plot
    plot_acf(series, ax=axes[0], lags=50)
    axes[0].set_title("ACF of Yahoo Stock")

    # PACF Plot
    plot_pacf(series, ax=axes[1], lags=50, method="ywm")
    axes[1].set_title("PACF of Yahoo Stock")

    st.pyplot(fig)

    # Forecasting model selection
    st.subheader("Forecasting Model")
    model_choice = st.selectbox("Select a forecasting model:", ["SARIMA", "ETS", "Prophet", "LSTM"])
    forecast_horizon = st.slider("Forecast Horizon (days):", min_value=1, max_value=60, value=30)

    if model_choice in ["SARIMA", "ETS", "Prophet"]:
        train, test = ts[:-forecast_horizon], ts[-forecast_horizon:]

    if st.button("Run Forecast"):

        if model_choice == "SARIMA":
            model = SARIMAX(train, order=(7, 1, 2), seasonal_order=(1, 1, 1, 30))
            model_fit = model.fit()
            forecast = model_fit.predict(start=len(train), end=len(train) + forecast_horizon - 1)
            forecast.index = test.index

        elif model_choice == "ETS":
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=30)
            model_fit = model.fit()
            forecast = model_fit.forecast(forecast_horizon)

        elif model_choice == "Prophet":
            prophet_df = ts.reset_index().rename(columns={"date": "ds", "adjClose": "y"})
            m = Prophet(growth='flat', daily_seasonality=True)
            m.fit(prophet_df[:-forecast_horizon])
            future = m.make_future_dataframe(periods=forecast_horizon)
            forecast_df = m.predict(future)
            forecast = forecast_df['yhat'].iloc[-forecast_horizon:].values

        elif model_choice == "LSTM":
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(ts.values.reshape(-1, 1))
            train_size = len(scaled_data) - forecast_horizon
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size - 60:]

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i - seq_length:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)

            seq_length = 60
            X_train, y_train = create_sequences(train_data, seq_length)
            X_test, y_test = create_sequences(test_data, seq_length)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

            predicted = model.predict(X_test)
            forecast = scaler.inverse_transform(predicted)[:forecast_horizon, 0]

        # Evaluation
        if model_choice != "LSTM":
            actual = test.values
        else:
            actual = ts[-forecast_horizon:].values

        rmse = round(np.sqrt(mean_squared_error(actual, forecast)), 2)
        mae = round(mean_absolute_error(actual, forecast), 2)
        mse = round(mean_squared_error(actual, forecast), 2)
        mape = round(np.mean(np.abs((actual - forecast) / actual)) * 100, 2)

        st.metric("RMSE", f"{rmse}")
        st.metric("MAE", f"{mae}")
        st.metric("MSE", f"{mse}")
        st.metric("MAPE", f"{mape}%")

        # Plot results
        st.subheader("Forecast vs Actual")
        fig, ax = plt.subplots()
        ax.plot(test.index if model_choice != "LSTM" else ts.index[-forecast_horizon:], actual, label='Actual')
        ax.plot(test.index if model_choice != "LSTM" else ts.index[-forecast_horizon:], forecast, linestyle='--', label='Forecast')
        ax.set_title(f"{model_choice} Forecast")
        ax.legend()
        st.pyplot(fig)