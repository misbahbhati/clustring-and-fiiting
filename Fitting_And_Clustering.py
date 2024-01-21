import pandas as pd
import sklearn.cluster as cl
import sklearn.preprocessing as pp

import matplotlib.pyplot as plt
import matplotlib


def merge_indicators():
    # Note: the top 4 lines were deleted by hand and the year columns which has no values
    Annual_Water = pd.read_csv("Ann_Fresh_Water_BCM.csv")

    # Note: the top 4 lines were deleted by hand and the year columns which has no values
    Green_House = pd.read_csv("GreenHouse_Gas_Emission.csv")


    # creating a new dataframe with the country name and data for 2020
    both = Annual_Water[["Country Name", "2020"]].copy()

    # renaming the data column
    both = both.rename(columns={"2020": "Fresh Water"})


    # Now we can simply copy a column from the file with the other indicator
    both["Green House Gas"] = Green_House["2020"]


    # Dropping Null values
    both = both.dropna()

    return both


both = merge_indicators()
both

# Nornalise the data by using a robust scaler

# Setting up the scaler
scaler = pp.RobustScaler()

# extract the columns to be used for clustering
to_clust = both[["Fresh Water", "Green House Gas"]]

# Fitting
scaler.fit(to_clust)

# Transforming
normalised_data = scaler.transform(to_clust)


# setting up the clustering function
nclust = 4
clusters = cl.KMeans(n_clusters=nclust, n_init=20)

# Doing the clustering. The result are stored in clusters
clusters.fit(normalised_data)

# Extracting the labels, i.e. the cluster nmumber
labels = clusters.labels_
print(labels)

# Extracting the cluster centres.
centres = clusters.cluster_centers_
centres = scaler.inverse_transform(centres)

# centres is a list of x and y values. Extract x and y.
xcen = centres[:, 0]
ycen = centres[:, 1]

print(centres)
print(xcen)
print(ycen)


# Selecting a colour map with high contrast.
cm = matplotlib.colormaps["Paired"]
plt.scatter(both["Fresh Water"], both["Green House Gas"], 10, labels, marker="o", cmap=cm)

# For the centres only one colour is selected
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.title('Fresh Water Vs Green House Gas Emission In 2020')
plt.xlabel("Fresh Water Usage (Billion Cubic Ton)")
plt.ylabel("Green House Gas Emition (KT CO2 Equivalantt)")
plt.show()


# adding the cluster memebership information to the dataframe
both["labels"] = labels
print(both)

both.labels.value_counts()

# Saving our clustered dataframe
both.to_excel("cluster_results.xlsx")


# !-----Fitting And Future Prediction-----!
# Selecting different indicator datasets for fitting
import pandas as pd
import matplotlib.pyplot as plt

freshwater_data = pd.read_csv('Ann_Fresh_Water_BCM.csv')
emission_data = pd.read_csv('GreenHouse_Gas_Emission.csv')


# Visualising trends over years

# Selecting the country
country_name = 'South Africa'

# Filtering data for the selected country
country_freshwater = freshwater_data[freshwater_data['Country Name'] == country_name]
country_emission = emission_data[emission_data['Country Name'] == country_name]

# Selecting years for plotting
selected_years = ['1990', '1995', '2000', '2005', '2010', '2015', '2020']

# Filtering data for selected years
country_freshwater_selected = country_freshwater[['Country Name'] + selected_years]
country_emission_selected = country_emission[['Country Name'] + selected_years]

# Plotting
plt.figure(figsize=(12, 6))

# Plotting Fresh Water Usage
plt.subplot(1, 2, 1)
plt.plot(selected_years, country_freshwater_selected.iloc[0, 1:], marker='o')
plt.title(f'{country_name} - Fresh Water Usage (Selected Years)')
plt.xlabel('Year')
plt.ylabel('Fresh Water Usage (billion cubic meters)')
plt.grid(True)

# Plotting Greenhouse Gas Emissions
plt.subplot(1, 2, 2)
plt.plot(selected_years, country_emission_selected.iloc[0, 1:], marker='o', color='orange')
plt.title(f'{country_name} - Greenhouse Gas Emission (Selected Years)')
plt.xlabel('Year')
plt.ylabel('Greenhouse Gas Emission (kt of CO2 equivalent)')
plt.grid(True)

plt.tight_layout()
plt.show()


# Fitting and predicting South Africa Fresh Water and Emission  - by using curve_fit method, quadratic model
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Selecting the columns for modeling
years = np.array(country_freshwater.columns[4:].astype(int))
freshwater_values = np.array(country_freshwater.iloc[0, 4:].astype(float))
emission_values = np.array(country_emission.iloc[0, 4:].astype(float))


# Defining a simple quadratic model
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c


# Fiting the quadratic model to Fresh Water Usage
freshwater_params, _ = curve_fit(quadratic_model, years, freshwater_values)
# Fiting the quadratic model to Greenhouse Gas Emission
emission_params, _ = curve_fit(quadratic_model, years, emission_values)


# Predicting future values till year 2030
future_years = np.array([2025, 2030])
freshwater_predictions = quadratic_model(future_years, *freshwater_params)
emission_predictions = quadratic_model(future_years, *emission_params)


# Plotting the original data and predictions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(years, freshwater_values, label='Actual Fresh Water Usage', marker='o')
plt.plot(future_years, freshwater_predictions, label='Predicted Fresh Water Usage', linestyle='--', marker='o')
plt.title(f'{country_name} - Fresh Water Usage Prediction')
plt.xlabel('Year')
plt.ylabel('Fresh Water Usage (billion cubic meters)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(years, emission_values, label='Actual Greenhouse Gas Emission', marker='o', color='orange')
plt.plot(future_years, emission_predictions, label='Predicted Greenhouse Gas Emission', linestyle='--', marker='o', color='orange')
plt.title(f'{country_name} - Greenhouse Gas Emission Prediction')
plt.xlabel('Year')
plt.ylabel('Greenhouse Gas Emission (kt of CO2 equivalent)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# # Prediction and Confidence range by using clustered data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Function to fit (simple linear model)
def simple_model(x, a, b):
    return a * x + b

# Function to estimate confidence intervals
def err_ranges(popt, pcov, x, conf_interval=0.95):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf((1 - conf_interval) / 2))
    upper_bound = simple_model(x, *(popt + z_score * perr))
    lower_bound = simple_model(x, *(popt - z_score * perr))
    return lower_bound, upper_bound

# Loading the cluster results dataframe
df = pd.read_excel("cluster_results.xlsx")

# Separating data by clusters
cluster_data = [df[df['labels'] == label] for label in df['labels'].unique()]

# Plotting the data, best-fitting function, and confidence intervals for each cluster
plt.figure(figsize=(12, 8))

for idx, cluster_df in enumerate(cluster_data):
    x_data = cluster_df['Fresh Water']
    y_data = cluster_df['Green House Gas']

    # Fitting the model to the data
    popt, pcov = curve_fit(simple_model, x_data, y_data)

    # Generating predictions for future values (including 10 and 20 years into the future)
    future_x = np.linspace(x_data.min(), x_data.max() + 20, 100)
    predicted_y = simple_model(future_x, *popt)

    # Estimating confidence intervals
    lower_bound, upper_bound = err_ranges(popt, pcov, future_x)

    # Plotting the data, best-fitting function, and confidence intervals
    plt.scatter(x_data, y_data, label=f'Cluster {idx}')
    plt.plot(future_x, predicted_y, label=f'Best-Fitting Function - Cluster {idx}')
    plt.fill_between(future_x, lower_bound, upper_bound, color='gray', alpha=0.2, label=f'Confidence Interval - Cluster {idx}')

plt.xlabel('Fresh Water')
plt.ylabel('Green House Gas Emission')
plt.legend()
plt.title('Fitting a Simple Model to Data with Confidence Intervals - Different Clusters')
plt.show()