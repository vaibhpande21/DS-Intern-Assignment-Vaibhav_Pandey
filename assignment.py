# %% [markdown]
# ## Dataset Overview
#
# The dataset contains sensor measurements collected from a manufacturing facility over time. It includes environmental readings from multiple zones within the factory, external weather conditions, and energy consumption metrics. The primary target variable is `equipment_energy_consumption`, which represents the energy used by manufacturing equipment in Wh (watt-hours).
#
# ## Feature Descriptions
#
# ### Time Information
# - `timestamp`: Date and time of the measurement (format: YYYY-MM-DD HH:MM:SS)
#
# ### Energy Metrics
# - `equipment_energy_consumption`: Energy consumption of manufacturing equipment in Wh (target variable)
# - `lighting_energy`: Energy consumption of lighting systems in Wh
#
# ### Zone Measurements
# The factory is divided into 9 distinct zones, each equipped with temperature and humidity sensors:
#
# #### Zone 1 - Main Production Area
# - `zone1_temperature`: Temperature in Zone 1 (°C)
# - `zone1_humidity`: Relative humidity in Zone 1 (%)
#
# #### Zone 2 - Assembly Line
# - `zone2_temperature`: Temperature in Zone 2 (°C)
# - `zone2_humidity`: Relative humidity in Zone 2 (%)
#
# #### Zone 3 - Quality Control
# - `zone3_temperature`: Temperature in Zone 3 (°C)
# - `zone3_humidity`: Relative humidity in Zone 3 (%)
#
# #### Zone 4 - Packaging Area
# - `zone4_temperature`: Temperature in Zone 4 (°C)
# - `zone4_humidity`: Relative humidity in Zone 4 (%)
#
# #### Zone 5 - Raw Material Storage
# - `zone5_temperature`: Temperature in Zone 5 (°C)
# - `zone5_humidity`: Relative humidity in Zone 5 (%)
#
# #### Zone 6 - Loading Bay
# - `zone6_temperature`: Temperature in Zone 6 (°C)
# - `zone6_humidity`: Relative humidity in Zone 6 (%)
#
# #### Zone 7 - Office Space
# - `zone7_temperature`: Temperature in Zone 7 (°C)
# - `zone7_humidity`: Relative humidity in Zone 7 (%)
#
# #### Zone 8 - Control Room
# - `zone8_temperature`: Temperature in Zone 8 (°C)
# - `zone8_humidity`: Relative humidity in Zone 8 (%)
#
# #### Zone 9 - Staff Area
# - `zone9_temperature`: Temperature in Zone 9 (°C)
# - `zone9_humidity`: Relative humidity in Zone 9 (%)
#
# ### External Weather Conditions
# - `outdoor_temperature`: Outside temperature (°C)
# - `outdoor_humidity`: Outside relative humidity (%)
# - `atmospheric_pressure`: Atmospheric pressure (mm Hg)
# - `wind_speed`: Wind speed (m/s)
# - `visibility_index`: Visibility measure (km)
# - `dew_point`: Dew point temperature (°C)
#
# ### Additional Variables
# - `random_variable1`: Random variable generated for the dataset
# - `random_variable2`: Random variable generated for the dataset
#
# ## Target Variable Details
#
# The main target variable is `equipment_energy_consumption`, which represents the energy consumption of manufacturing equipment measured in watt-hours (Wh). This is a continuous variable that typically ranges from 10 to 1080 Wh in the dataset. The distribution is right-skewed, with most values concentrated in the lower range.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import calendar

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, ReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

import warnings

warnings.filterwarnings("ignore")

# %%
## load data

df = pd.read_csv(
    "/Users/phoenix/Documents/GitHub/DS-Intern-Assignment-Vaibhav_Pandey/data/data.csv"
)
df.head()

# %% [markdown]
# #### Data Preprocessing

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df.dtypes

# %%
# Convert specified columns to float

for col in [
    "equipment_energy_consumption",
    "lighting_energy",
    "zone1_temperature",
    "zone1_humidity",
    "zone2_temperature",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# %%
df.dtypes

# %%
## Dropping rows with negative values in 'equipment_energy_consumption' and 'lighting_energy'

df = df[~(df["lighting_energy"] < 0)]
df = df[~(df["equipment_energy_consumption"] < 0)]

# %% [markdown]
# #### I noticed that there were some negative entries in 'equipment_energy_consumption' and 'lighting_energy', since these are units of power in Wh. It implies that there is some error in capturing the correct reading from the sensor hence i decided to remove them completelyy from the dataset.

# %%
df.shape

# %%
humidity_columns = [
    col for col in df.columns if "humidity" in col.lower() and col != "outdoor_humidity"
]

# Combine all humidity columns (including outdoor_humidity)
all_humidity_columns = humidity_columns + ["outdoor_humidity"]

invalid_humidity_mask = df[all_humidity_columns].lt(0).any(axis=1)

affected_rows_count = invalid_humidity_mask.sum()

# %%
print(f"Affected rows: {affected_rows_count}")
print(f"Percentage of dataset affected: {affected_rows_count / len(df) * 100:.2f}%")


# %%
def drop_negative_humidity(df):
    """
    Function to drop rows with negative humidity values.
    Action:
    Drop rows with negative humidity values in any humidity-related column.
    """

    humidity_cols = [col for col in df.columns if "humidity" in col.lower()]
    mask = (df[humidity_cols] < 0).any(axis=1)
    print(f"Dropping {mask.sum()} rows with negative humidity values.")

    df_cleaned = df[~mask].reset_index(drop=True)
    return df_cleaned


# %%
df = drop_negative_humidity(df)
df.shape

# %% [markdown]
# #### I noticed that there were some negative entries in the humidity-related columns, including 'zone1_humidity' to 'zone9_humidity' and 'outdoor_humidity'. Since humidity is a percentage and cannot logically take negative values, this implies an error in capturing the correct readings from the sensors. Upon analysis, I found that 2,337 rows (14.32% of the dataset) had at least one invalid humidity value. To ensure data quality and avoid introducing bias through imputation, I decided to remove these rows entirely from the dataset.

# %%
df.isnull().sum()

# %%
## Convert the 'timestamp' column to datetime format

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(df.dtypes)
df.head()

# %%
#  Drop rows with missing target

df = df.dropna(subset=["equipment_energy_consumption"])

# %%
df.isnull().sum()

# %% [markdown]
# ### Handling Missing Values

# %%
# Impute missing values

for col in df.columns:
    if col != "equipment_energy_consumption" and df[col].isnull().any():
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)

# %%
df.isnull().sum()

# %% [markdown]
# ### Feature Engineering

# %%
# Extracting date and time features from the timestamp

df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["hour"] = df["timestamp"].dt.hour
df.head()

# %%
# Drop the original timestamp column

df.drop(["timestamp"], axis=1, inplace=True)
df.head()

# %%
## create new features based on zone temperature and humidity

zone_temp_cols = []
zone_humid_cols = []

for col in df.columns:
    if "zone" in col and "temperature" in col:
        zone_temp_cols.append(col)
    elif "zone" in col and "humidity" in col:
        zone_humid_cols.append(col)

# create new features based on temperature and humidity

df["mean_zone_temperature"] = df[zone_temp_cols].mean(axis=1)
df["mean_zone_humidity"] = df[zone_humid_cols].mean(axis=1)
df["temperature_diff"] = df["outdoor_temperature"] - df["mean_zone_temperature"]
df["humidity_diff"] = df["outdoor_humidity"] - df["mean_zone_humidity"]

df["zone_temperature_std"] = df[zone_temp_cols].std(axis=1)
df["zone_humidity_std"] = df[zone_humid_cols].std(axis=1)
df["zone_temperature_range"] = df[zone_temp_cols].max(axis=1) - df[zone_temp_cols].min(
    axis=1
)

# %%
# Weather interactions

df["temp_humidity_interaction"] = df["outdoor_temperature"] * df["outdoor_humidity"]

# %%
# Lag features (using 1-hour lag)

df["energy_lag_1"] = df["equipment_energy_consumption"].shift(1)

# %%
# Rolling statistics

df["rolling_3h_mean"] = df["equipment_energy_consumption"].rolling(window=3).mean()

# %%
# Handle missing values

df["energy_lag_1"].fillna(method="bfill", inplace=True)
df["rolling_3h_mean"].fillna(method="bfill", inplace=True)

# %%
df.head()

# %% [markdown]
# ### Exploratory Data Analysis

# %%
## Checking distribution of target variable

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(
    df["equipment_energy_consumption"], bins=30, color="skyblue", edgecolor="black"
)
plt.title("Distribution of Equipment Energy Consumption")
plt.xlabel("Equipment Energy Consumption")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.kdeplot(df["equipment_energy_consumption"], color="red", shade=True)
plt.title("KDE of Equipment Energy Consumption")
plt.xlabel("Equipment Energy Consumption")
plt.show()

# %% [markdown]
# #### Energy consumption is right-skewed, with most values concentrated below 150 Wh. A few outliers extend beyond 1000 Wh.

# %%
## KDE plots for outdoor humidity and temperature

plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")
days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Plot 1: Outdoor Humidity vs Day of Week
plt.subplot(1, 2, 1)
sns.kdeplot(
    data=df, x="dayofweek", y="outdoor_humidity", color="mediumturquoise", fill=True
)
ax = plt.gca()
ax.set_xticks(range(7))
ax.set_xticklabels(days)
plt.title("Outdoor Humidity by Day of Week")

# Plot 2: Outdoor Humidity vs Hour of Day
plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x="hour", y="outdoor_humidity", color="mediumturquoise", fill=True)
plt.title("Outdoor Humidity by Hour of Day")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Outdoor humidity remains fairly stable across the week. Hourly patterns show higher humidity early in the morning (around 5–7 AM) and lower levels during afternoon hours.

# %%
## Grouping by day of week and hour to get mean outdoor humidity

humidity_by_day = df.groupby("dayofweek")["outdoor_humidity"].mean()
humidity_by_hour = df.groupby("hour")["outdoor_humidity"].mean()

# %%
## Plotting mean outdoor humidity by day of week and hour

plt.figure(figsize=(15, 5))

# Mean Outdoor Humidity by Day of Week
plt.subplot(1, 2, 1)
sns.lineplot(x=days, y=humidity_by_day.values, marker="o")
plt.title("Avg Outdoor Humidity by Day of Week")

# Mean Outdoor Humidity by Hour
plt.subplot(1, 2, 2)
sns.lineplot(x=humidity_by_hour.index, y=humidity_by_hour.values, marker="o")
plt.title("Avg Outdoor Humidity by Hour")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Outdoor humidity is highest early in the morning (5–7 AM) and lowest in the afternoon (2–3 PM). Weekly variation is minimal, with slightly higher humidity on Saturday and Monday. These time-based patterns could help improve energy consumption predictions.

# %%
## Convert month number to month name

df["month_name"] = df["month"].apply(lambda x: calendar.month_name[x])

# %%
## Grouping by month name to get mean outdoor humidity

monthly_avg_humidity = df.groupby("month")["outdoor_humidity"].mean()
month_names = [calendar.month_name[i] for i in monthly_avg_humidity.index]

# %%
## Plotting mean outdoor humidity by month

plt.figure(figsize=(10, 5))
sns.lineplot(x=month_names, y=monthly_avg_humidity.values, marker="o", color="teal")
plt.title("Average Outdoor Humidity by Month")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Humidity generally declines from January to July, then spikes sharply in September, likely due to seasonal weather patterns.

# %%
## Grouping by month to get mean outdoor temperature

monthly_avg_temp = df.groupby("month")["outdoor_temperature"].mean()
month_names = [calendar.month_name[i] for i in monthly_avg_temp.index]

# %%
## plotting mean outdoor temperature by month

plt.figure(figsize=(10, 5))
sns.lineplot(x=month_names, y=monthly_avg_temp.values, marker="o", color="coral")
plt.title("Average Outdoor Temperature by Month")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Outdoor temperature rises steadily from January to July, then drops sharply in September, followed by a quick recovery in October.

# %%
## Grouping by day of week and hour to get mean outdoor temperature

hourly_avg_temp = df.groupby("hour")["outdoor_temperature"].mean()
avg_temp_by_day = df.groupby("dayofweek")["outdoor_temperature"].mean()

# %%
## Plotting average outdoor temperature by day of week and hour

plt.figure(figsize=(15, 5))

# Mean Outdoor Temp by Day of Week
plt.subplot(1, 2, 1)
sns.lineplot(x=days, y=avg_temp_by_day.values, marker="o", color="orange")
plt.title("Avg Outdoor Temp by Day of Week")

# Mean Outdoor Temp by Hour
plt.subplot(1, 2, 2)
sns.lineplot(
    x=hourly_avg_temp.index, y=hourly_avg_temp.values, marker="o", color="orange"
)
plt.title("Avg Outdoor Temp by Hour")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### By Day: Temps are coolest on Monday, peaking on Friday.
# #### By Hour: Temperature rises after 9 AM, peaking between 2–4 PM, then gradually drops.

# %%
## Plotting Correlation Heatmap

plt.figure(figsize=(18, 15))
corr = df.corr(numeric_only=True)

# Annotate with correlation scores
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="lightgray",
)

plt.title("Correlation Heatmap (with Coefficients)", fontsize=16)
plt.xticks()
plt.yticks()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Decision Regarding 'random_varibale1' and 'random_varibale2'
#
# We can observe from the correlation matrix that both of these variables do not have significant correlation with any of the other variables meaning that they are not carrying redundant information. However i decided to futher investigate the affect of these two variables in predicting the target variable using an appropriate hypothesis test.

# %% [markdown]
# ### Linear Regression Coefficient t-test
#
# Fitting a multiple linear regression model:
#
# H0 (null hypothesis): regression coefficient of predictor variable(beta) = 0 (the variable has no effect)
#
# H1 (alternative hypothesis): regression coefficient of predictor variable(beta) <> 0 (the variable significantly affects energy consumption)
#
# Interpret the p-values:
#
# If p-value < 0.05, reject H₀ : the variable is significantly predictive.

# %%
X = df[["random_variable1", "random_variable2"]]
y = df["equipment_energy_consumption"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

# %% [markdown]
# ### Analyzing the result:
#
# R-squared = 0.000: The model explains 0% of the variance in 'equipment_energy_consumption'. This means the predictors have no explanatory power.
#
# F-statistic = 1.934, p-value (Prob F-statistic) = 0.145:
#
# This tests the overall significance of the regression.
#
# Since p > 0.05, the model is not statistically significant overall.

# %% [markdown]
# ### Conclusion:
#
# beta1 (random_variable1) = –0.0728 : means a small negative association, but not statistically significant (p = 0.165)
#
# beta2 (random_variable2) = –0.0517 : also small and not significant (p = 0.337)
#
# Both p-values are greater than 0.05, so we fail to reject the null hypothesis: neither variable has a statistically significant effect on the target.
#
# Hence I decided to remove both the columns.

# %%

df.drop(["random_variable1", "random_variable2", "month_name"], axis=1, inplace=True)

# %%
df.drop(
    [
        "temp_humidity_interaction",
        "temperature_diff",
        "humidity_diff",
        "zone_temperature_range",
    ],
    axis=1,
    inplace=True,
)

# %% [markdown]
# ####  Dropping redundant features which shows high correlation among themselves, hence of no use in our model building.

# %%
# Remove duplicate rows

df.drop_duplicates(inplace=True)

# %% [markdown]
# ### Outlier detection and treatment

# %%
### Checking for outliers

num_cols = [
    "equipment_energy_consumption",
    "lighting_energy",
    "outdoor_temperature",
    "outdoor_humidity",
    "atmospheric_pressure",
    "wind_speed",
    "visibility_index",
    "dew_point",
]

fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(15, 15))

for i, col in enumerate(num_cols):
    sns.distplot(df[col], ax=ax[i, 0], color="mediumturquoise")
    sns.boxplot(data=df, x=df[col], ax=ax[i, 1], color="#E6A9EC")
    ax[i, 0].set_title(f"{col} Distribution")
    ax[i, 1].set_title(f"Boxplot of {col}")


plt.tight_layout()

# %%
numeric_features = df.select_dtypes(include="number").columns

# Outlier detection using IQR method
outlier_summary = []

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary.append(
        {
            "Feature": col,
            "Num_Outliers": len(outliers),
            "Percent_Outliers": round(100 * len(outliers) / len(df), 2),
        }
    )


outlier_df = pd.DataFrame(outlier_summary).sort_values(
    by="Percent_Outliers", ascending=False
)
outlier_df.reset_index(drop=True, inplace=True)
outlier_df.head(15)

# %%
clip_features = ["lighting_energy", "visibility_index"]

for col in clip_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# %% [markdown]
# #### Identified and clipped outliers for features where more than 15% of the values lie outside the typical range (e.g., beyond 1.5×IQR from Q1/Q3). This was done to reduce the influence of extreme values without removing significant portions of the data.

# %%
df.head()

# %% [markdown]
# ### Scaling & Train-Test Split

# %%
X = df.drop("equipment_energy_consumption", axis=1)
y = df["equipment_energy_consumption"]

# %%
X.shape

# %%
y.shape

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.25, random_state=42
)

# %%
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
X_test = scaler.transform(X_test)

# %% [markdown]
# ### Model Building


# %%
def evaluate_model(name, model, x_val, y_val):
    """
    Function to evaluate the model on validation data.
    """
    y_pred = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"\n{name} Validation Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    return rmse, mae, r2


# %% [markdown]
# #### Linear Regression Model

# %%
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# %%
evaluate_model("Linear Regression", lr_model, x_val, y_val)

# %% [markdown]
# #### Linear Regression with L2 regularisation

# %%
ridge = Ridge(alpha=0.01)
ridge.fit(x_train, y_train)

# %%
evaluate_model("Ridge Regression", ridge, x_val, y_val)

# %% [markdown]
# #### Linear Regression using Polynomial Features

# %%
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_val_poly = poly.transform(x_val)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)

# %%
evaluate_model("Polynomial Regression (Degree=2)", poly_model, x_val_poly, y_val)

# %% [markdown]
# #### Linear Regression using PCA

# %%
pca = PCA(n_components=25)
x_train_pca = pca.fit_transform(x_train)
x_val_pca = pca.transform(x_val)

lr_pca_model = LinearRegression()
lr_pca_model.fit(x_train_pca, y_train)

# %%
evaluate_model("Linear Regression with PCA", lr_pca_model, x_val_pca, y_val)
print("→ Number of Components Used:", pca.n_components_)

# %%
dt_params = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# %%
dt_model = DecisionTreeRegressor(random_state=42)

dt_grid = GridSearchCV(
    estimator=dt_model, param_grid=dt_params, cv=3, n_jobs=-1, scoring="r2", verbose=2
)
dt_grid.fit(x_train, y_train)

# %%
evaluate_model("Decision Tree (Tuned)", dt_grid.best_estimator_, x_val, y_val)

# %% [markdown]
# #### XGBosst

# %%
xgb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# %%
xgb_model = XGBRegressor(random_state=42, objective="reg:squarederror")
xgb_grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_params,
    cv=3,
    n_jobs=-1,
    scoring="neg_mean_squared_error",
    verbose=2,
)
xgb_grid.fit(x_train, y_train)

# %%
evaluate_model("XGBoost (Tuned)", xgb_grid.best_estimator_, x_val, y_val)

# %% [markdown]
# #### Neural Network

# %%
model = Sequential(
    [
        Dense(64, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.3),
        Dense(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        ReLU(),
        Dense(1, activation="linear"),
    ]
)

# %%
model.summary()

# %%
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=[MeanSquaredError(name="mse"), MeanAbsoluteError(name="mae")],
)

# %%
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,  # Wait 10 epochs after val loss stops improving
    restore_best_weights=True,
)

# %%
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,  # Reduce LR by 80% when plateau
    patience=5,
    min_lr=1e-6,
)

# %%
ModelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model_reg.h5", save_best_only=True
)

# %%
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, ModelCheckpointCallback],
    verbose=1,
)

# %%
epochs = history.epoch
loss = history.history["loss"]
val_loss = history.history["val_loss"]

mae = history.history["mae"]
val_mae = history.history["val_mae"]

# %%
plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label="train")
plt.plot(epochs, val_loss, label="val")
plt.legend()
plt.title("Loss VS Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, mae, label="train")
plt.plot(epochs, val_mae, label="validation")
plt.legend()
plt.title("Mean Absolute Error VS Epochs")
plt.xlabel("Epochs")
plt.ylabel("MAE")

plt.tight_layout()
plt.show()

# %%
model.save("best_model_reg.h5")

# %%
y_pred = model.predict(x_val)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Print results
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# %% [markdown]
# #### Support Vector Machine

# %%
svr_model = SVR(kernel="rbf", C=100, epsilon=0.1)
svr_model.fit(x_train, y_train)

# %%
evaluate_model("SVR", svr_model, x_val, y_val)

# %% [markdown]
# ### Evaluating test dataset with best model based on validation results.

# %%
y_lr_pred = lr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_lr_pred))
mae_lr = mean_absolute_error(Y_test, y_lr_pred)
r2_lr = r2_score(Y_test, y_lr_pred)

print(f"Linear Regression Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")

# %%
models = [
    "Linear Regression",
    "Ridge Regression",
    "Polynomial Regression (Deg=2)",
    "Linear Regression + PCA",
    "Decision Tree (Tuned)",
    "XGBoost (Tuned)",
    "Neural Network",
    "SVR",
]

# %%
rmse = [92.66, 92.66, 103.06, 120.83, 105.39, 94.27, 94.2393, 98.66]
mae = [35.26, 35.26, 43.23, 47.70, 40.91, 34.24, 36.9918, 35.78]
r2 = [0.59, 0.59, 0.49, 0.30, 0.47, 0.58, 0.5754, 0.53]

df = pd.DataFrame({"Model": models, "RMSE": rmse, "MAE": mae, "R2 Score": r2})

# %%
# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
df.plot.bar(x="Model", y="RMSE", ax=axes[0], legend=False, color="skyblue")
axes[0].set_title("Validation RMSE by Model")
axes[0].set_ylabel("RMSE")
axes[0].grid(True, linestyle="--", alpha=0.7)

df.plot.bar(x="Model", y="MAE", ax=axes[1], legend=False, color="lightgreen")
axes[1].set_title("Validation MAE by Model")
axes[1].set_ylabel("MAE")
axes[1].grid(True, linestyle="--", alpha=0.7)

df.plot.bar(x="Model", y="R2 Score", ax=axes[2], legend=False, color="salmon")
axes[2].set_title("Validation R² Score by Model")
axes[2].set_ylabel("R² Score")
axes[2].grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
#  ### Model Performance Summary Report | Validation Set Evaluation

# %% [markdown]
# | Model                             | RMSE   | MAE   | R² Score |
# | --------------------------------- | ------ | ----- | -------- |
# | **Linear Regression**             | 92.66  | 35.26 | 0.59     |
# | **Ridge Regression**              | 92.66  | 35.26 | 0.59     |
# | **Polynomial Regression (Deg=2)** | 103.06 | 43.23 | 0.49     |
# | **Linear Regression + PCA**       | 120.83 | 47.70 | 0.30     |
# | **Decision Tree (Tuned)**         | 105.39 | 40.91 | 0.47     |
# | **XGBoost (Tuned)**               | 94.27  | 34.24 | 0.58     |
# | **Neural Network**                | 94.47  | 37.12 | 0.5754   |
# | **SVR**                           | 98.66  | 35.78 | 0.53     |

# %% [markdown]
# ### Test Set Evaluation (Best Model: Linear Regression)

# %% [markdown]
# | Metric   | Value  |
# | -------- | ------ |
# | **RMSE** | 91.68  |
# | **MAE**  | 35.72  |
# | **R²**   | 0.5698 |

# %% [markdown]
# ### Conclusion :
# Linear Regression and Ridge Regression achieved the best overall performance on the validation set, with the lowest RMSE (92.66) and highest R² score (0.59).
#
# XGBoost had the lowest MAE (34.24), making it a strong contender in terms of error minimization.
#
# Polynomial Regression and PCA-based models performed poorly, likely due to overfitting or information loss.
#
# The final test set evaluation using Linear Regression confirms consistent performance, with RMSE = 91.68 and R² ≈ 0.57.
