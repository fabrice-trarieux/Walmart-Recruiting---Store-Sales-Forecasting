import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF

def plot_dist(df, value, fit_norm=False, fig_size=(20,15), groupby=None):
    y = value
    x = groupby

    fig, ax = plt.subplots(2,2, figsize=fig_size)
    # violin plot
    plt.subplot(3, 2, 1);
    sns.violinplot(data=df, y=y, x=x);
    # box plot
    plt.subplot(3, 2, 2);
    sns.boxplot(data=df, y=y, x=x);
    if x is None:
        # ecdf plot
        plt.subplot(3, 2, 3);
        ecdf = ECDF(df[y])
        ecdf_x, ecdf_y = ecdf.x, ecdf.y
        plt.plot(ecdf_x, ecdf_y, marker='.', linestyle='none', label=x);
        if fit_norm:
            sample = np.random.normal(np.mean(df[y]), np.std(df[y]), size=df[y].shape[0])
            ecdf = ECDF(sample)
            ecdf_x, ecdf_y = ecdf.x, ecdf.y
            plt.plot(ecdf_x, ecdf_y, marker='.', linestyle='none', label=x);
        # density plot (pdf)
        plt.subplot(3, 2, 4);
        sns.distplot(df[y], norm_hist=True);
        if fit_norm:
            sns.distplot(sample, norm_hist=True, hist_kws=dict(alpha=0.05));
    else:
        # ecdf plot
        plt.subplot(3, 2, 3);
        for g in df[x].unique():
            ecdf = ECDF(df.loc[df[x]==g, y])
            ecdf_x, ecdf_y = ecdf.x, ecdf.y
            plt.plot(ecdf_x, ecdf_y, marker='.', linestyle='none', label=g);
        plt.subplot(3, 2, 4);
        for g in df[x].unique():
            sns.distplot(df.loc[df[x]==g, y], norm_hist=True);
    plt.suptitle(y)
    plt.show()

def plot_dist_2_samples(df1, df2, value, fig_size=(20,15)):
    y = value

    fig, ax = plt.subplots(2,2, figsize=fig_size)
    # violin plot
    plt.subplot(3, 2, 1);
    sns.violinplot(data=df1, y=y);
    sns.violinplot(data=df2, y=y);
    # box plot
    plt.subplot(3, 2, 2);
    sns.boxplot(data=df1, y=y);
    sns.boxplot(data=df2, y=y);
    # ecdf plot
    plt.subplot(3, 2, 3);
    ecdf = ECDF(df1[y])
    ecdf_x, ecdf_y = ecdf.x, ecdf.y
    plt.plot(ecdf_x, ecdf_y, marker='.', linestyle='none');
    ecdf = ECDF(df2[y])
    ecdf_x, ecdf_y = ecdf.x, ecdf.y
    plt.plot(ecdf_x, ecdf_y, marker='.', linestyle='none');
    # density plot (pdf)
    plt.subplot(3, 2, 4);
    sns.distplot(df1[y], norm_hist=True);
    sns.distplot(df2[y], norm_hist=True);  
    plt.suptitle(y)
    plt.show()