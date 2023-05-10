# sklearn, numpy imports

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from base_module import base_class

if __name__ == "__main__":
    try:
        file_name   = "climate_Change_Data.csv"
        years       = ['1990','2015']
        n_clusters  = 4

        data_df     = base_class.read_func(file_name)
        """  Can select the different kind of indicator here, but only one at a time. """
        indicator_selected_df   = base_class.indicator_selection_func(data_df = data_df, indicator_n = "Urban population growth (annual %)")
        years_selected_df       = base_class.columns_selected_func(indicator_selected_df, years)
        climate_change_df       = years_selected_df.dropna()  # Records with nan Values/nulls are useless

        base_class.data_plot_func(climate_change_df, years)

        normalize_data = base_class.normalizeDfFunc(climate_change_df)

        k_value_score, KMEANS_inter = base_class.loop_through_k_value_func(normalize_data, 10)
        base_class.elbow_plot_func(k_value_score, KMEANS_inter) # A New graph will open to Analyze the best k_value score selected indicator.

        kmeans_model = KMeans(n_clusters = 5) # for our Case the best k_value score is 4, thats why we just hardcode the value 5, BTW value could get change here.
        kmeans_model.fit(normalize_data)
        climate_change_df["predictions"] = kmeans_model.labels_

        base_class.clustered_plot_func(climate_change_df, years) # function will show and create the graph image for all the clusters in the selected indicator dataset.

        # curve_fit
        data_df = data_df
        indicatorSelectedDf = base_class.indicator_selection_func(data_df = data_df, indicator_n = "Urban population (% of total population)")
        selectedYears   = base_class.filterDataFunc(indicatorSelectedDf, "Country Name", "Arab World")
        x = selectedYears.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
        x = x.values.reshape((x.shape[1]))
        x = x[~np.isnan(x)]
        # parameters for the exponantial_growth
        a = 2
        b = 0.5
        c = 0
        y = base_class.exponential_growth(x, a, b, c)
        y = y[~np.isnan(y)]

        # popt > Optimal values for the parameters so that the sum of the squared residuals
        # pcov > The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        popt, pcov = curve_fit(base_class.exponential_growth, x, y)

        x_predictions = np.linspace(x.min(), x.max() + 10, 20)
        lower_bound, upper_bound = base_class.err_ranges(popt, pcov, x_predictions)

        # Plot the data, best fit, and confidence range
        base_class.curve_fit_plot_func(x, y, x_predictions, lower_bound, upper_bound, popt)
        
    except Exception as e:
        print("Process is stopped due to that reason:", e)
        print("Have to start Again the process")

