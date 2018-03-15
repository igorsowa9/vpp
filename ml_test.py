import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


if __name__ == '__main__':

    df = pd.read_csv("./Notebooks/MachineLearningWithPython/Notebooks/data/pima-data.csv")

    # print(df.shape)

    # print(df.head(5))
    # print(df.tail(5))

    #print(df.isnull().values.any())

    plot_corr(df)
    plt.show()

    print(df.corr())