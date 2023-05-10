# sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class base_class():

    def read_func(f_name):
        try:
            data_df = pd.read_csv(f_name)
        except Exception as e:
            print(e)
            print("Invalid Path")
            return None
        return data_df
    
    def indicator_selection_func(data_df, indicator_n):
        """
        this function is used to filter for the specific given indicator
        """
        selectIndicationDf = data_df[data_df['Indicator Name'] == indicator_n]
        shape = selectIndicationDf.shape
        if shape[0] == 0:
            print("Empty Dataframe! Please Enter the Valid indicator Name")
            return None
        else:
            print("indicator selected")
            return selectIndicationDf
    
    def filterDataFunc(data_df, column, value):
        filteredDf = data_df[data_df[f'{column}'] == value]
        shape = filteredDf.shape
        if shape[0] == 0:
            print("Empty Dataframe! Please Enter the Valid column Name or Enter the valid value")
            return None
        else:
            return filteredDf
    
    def columns_selected_func(data_df, years):
        """
        this function is used to filter and get the specific given columns
        """
        data_df = data_df[years]
        print("Columns selected for the given dataSet")
        return data_df
    
    def data_plot_func(data_df, years):
        """ 
        function will create plot will open and save the PNG image thats could be used to analyse the data distribution in the dataset.
        """
        plt.scatter(data_df[years[0]], data_df[years[1]])
        plt.xlabel(f"{years[0]}")
        plt.ylabel(f"{years[1]}")
        plt.title("data distribution across a particular indicator")
        plt.savefig("data visualize for a particular indicator")
        plt.show()
        print("Plot has been saved.")
        return True
        
    
    def normalizeDfFunc(data_df):
        scaler = StandardScaler()
        scaler.fit(data_df)
        # standardScaler is a way to preprocess the data and transform/normalize it between 1,-1. 1 means the highest value and -1 meand the lowest value
        scaled_data = scaler.transform(data_df)
        return scaled_data
    
    def loop_through_k_value_func(data_df, maximum_K):
        """
        function will loop through till maximum_k using the python's range function. 
        And will return all over the KMEANS intertia value k_value score for the kmeans models accordingly.
        """
        KMEANS_inter = []
        k_value_score = []
        
        for k in range(1, maximum_K):
            kmeans_model = KMeans(n_clusters = k)
            kmeans_model.fit(data_df)
            KMEANS_inter.append(kmeans_model.inertia_)
            k_value_score.append(k)
        return k_value_score, KMEANS_inter

    def elbow_plot_func(k_value_score, KMEANS_inter):
    
        figure = plt.subplots(figsize = (12, 6))
        plt.plot(k_value_score, KMEANS_inter, 'o-', color = 'blue')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("kmeans Inertia")
        plt.title("KMEANS elbowplot")
        plt.savefig("elbow_plot")
        plt.show()
        return True

    def clustered_plot_func(data_df, years):
        """ 
        function will create plot will open and save the clustered PNG image to get Insights the data 
        and compare the different clustered Values.
        """
        plt.scatter(data_df[[years[0]]], data_df[[years[1]]], c = data_df["predictions"])
        plt.xlabel(f"{years[0]}")
        plt.ylabel(f"{years[1]}")
        plt.title("clustered Image indicates the best data differences in the dataSet")
        plt.savefig("clustered_image")
        plt.show()

    def exponential_growth(x, a, b, c=0):
        # np.exp is used to get the exponential value for the array
        # np.random.normal used to get the Normal Distribution. It is one of the most important distributions that fits the probability distribution of many events, eg. IQ Scores, Heartbeat etc.
        return a * np.exp(b * x) + np.random.normal(0, 0.2, x.shape[0]) + c
    
    def err_ranges(popt, cov_matrix, x):
        errors = np.sqrt(np.diag(cov_matrix))
        lowerBound = base_class.exponential_growth(x, *(popt - errors))
        upperBound = base_class.exponential_growth(x, *(popt + errors))
        return lowerBound, upperBound
    
    def curve_fit_plot_func(x, y, x_predictions, lower_bound, upper_bound, popt):
            plt.scatter(x, y, label='actual data')
            plt.plot(x_predictions, base_class.exponential_growth(x_predictions, *popt), label='Best Fit')
            plt.fill_between(x_predictions, lower_bound, upper_bound, alpha=1, label='Confidence Range')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title("cruve_fit with future prediction")
            plt.legend()
            plt.savefig("curve_fit")
            plt.show()

