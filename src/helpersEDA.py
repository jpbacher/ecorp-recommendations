#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')



def datadict(df, feat):
    '''retrieves description of the feature from a data dictionary'''
    return df.loc[feat]['Description']



def get_duplicates(df, subset):
    
    print('The number of duplicates: {}'.format(sum(df.duplicated(subset=subset))))



def quick_observation(df, subset):
    
    """A quick look at certain characteristics of a dataset.
    
    Argument:
    df: the dataframe
    subset: a string, the columns to look for duplicate entries
    
    Returns:
    Prints various properties of the dataframe.
    """
    
    print('------The shape of the uncleaned training data: {}\n'.format(df.shape))
    print('------The first 5 rows:\n{}\n'.format(df.head()))
    print('------The datatypes of the predictors and target:\n{}\n'.format(df.dtypes))
    print('------The number of duplicates: {}'.format(sum(df.duplicated(subset=subset))))


    
def unique_values(df, tgt, high_card_num=None):
    '''look for variables that do not contain any variance and variables that
    have high cardinality, something we should be aware of....'''
    
    for col in cat_cols(df, tgt):
        if df[col].unique().size < 2:
            print('{} has 1 unique value'.format(col))
        elif df[col].unique().size >= high_card_num:
            print('Variable with high cardinality:\n{}---{}'.format(col,   df[col].unique().size))   
    
  

def cat_cols(df, tgt):  
    '''get a list of the categorical columns in a dataframe, excluding the target variable  '''
    
    cat_cols = df.dtypes[df.dtypes == 'object'].index
    return [col for col in cat_cols if tgt not in col]


def num_cols(df, tgt):  
    '''get a list of the numerical columns in a dataframe, excluding the target variable'''
    
    num_cols = df.dtypes[(df.dtypes == np.float) | (df.dtypes == np.integer)].index
    return [col for col in num_cols if tgt not in col]



def cat_mean(df, col, tgt):
    return (df.groupby(col)[tgt].mean()).sort_values(ascending=False)



def encode(df, col, tgt):
    '''encode a nominal variable with the mean of the target(for continuous targets) for each
    category of that variable'''
    
    cat_dict = dict()
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat][tgt].mean()
    df[col] = df[col].map(cat_dict)



def plt_tgt(df, tgt):
    
    '''creates a kdeplot & boxplot for the response variable; prints the p-value of the 
    normal test: the null hypothesis that a sample comes from a normal distribution'''
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    sns.kdeplot(df[tgt].dropna(), legend=False)
    sns.despine(left=True)
    plt.xlabel(tgt)
    plt.ylabel('Density')
    plt.title('%s Distribution' % str.title(tgt))
    
    plt.subplot(122)
    sns.boxplot(df[tgt], showmeans=True)
    sns.despine(left=True)
    plt.show()
    
    stat, p = stats.normaltest(df[tgt], nan_policy='omit')
    print('Normal test P-value: %.6f' % p)
    
    
    
def plt_tgt_classif(df, tgt):
    
    '''creates a simple bar chart of the response variable's value counts'''
    
    plt.figure(figsize=(12, 6))
    sns.barplot(df[tgt].value_counts(), df[tgt].dropna().unique(), orient='h')
    sns.despine(left=True)
    plt.xlabel('Count')
    plt.title('Distribution of %s' % str.capitalize(tgt))
    plt.show()
    print('Percentage of each class:\n{}'.format(df[tgt].value_counts() / df.shape[0]))
    
    
    
def iqr_outliers(df, col):
    '''Return the cutoff values of a particular predictor/response variable using 1.5 times the IQR'''

    percentiles = df[col].describe()

    IQR = percentiles['75%'] - percentiles['25%']
    upper = percentiles['75%'] + (IQR * 1.5)
    lower = percentiles['25%'] - (IQR * 1.5)
    return lower, upper



def outlier(df, col, n, upper=None, lower=None):
    '''Return dataframe of possible outliers for particular feature or target'''
    
    if upper: 
        return df.loc[df[col] > n]
    elif lower:
        return df.loc[df[col] < n]



def plt_feat(df, col, tgt, max_col_vals=None):
    
    '''Plot distributions of a feature; if continuous variable, plot data of the response variable 
    to that feature; if categorical, plot a boxplot of target variable to the nominal 
    variables' unique values
    Arguments:
    df: the dataframe
    col: the feature
    tgt: the response variable
    frac: the fraction of samples to use when plotting the data for a continuous feature
    max_col_vals: if a categorical variables' unique values above a threshold, we will not
    plot that feature'''
    
    if col == tgt:
        pass
    
    elif df[col].dtype == np.float or df[col].dtype == np.integer:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            sns.kdeplot(df[col].dropna(), color='Purple', legend=False)
            plt.xlabel(col)
            plt.ylabel('Density')
            sns.despine(left=True)
            plt.title('Distribution of %s' % str.capitalize(col))
    
            plt.subplot(122)
            sns.lineplot(df[col].dropna(), df[tgt].dropna(), color='Blue', ci=None)
            sns.despine(left=True)
            plt.title('%s to %s' % (str.capitalize(tgt), str.capitalize(col)))
            plt.show()
    
    elif df[col].dtype == 'object':
        
        if len(df[col].unique()) < max_col_vals:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            sns.barplot(df[col].dropna().unique(), df[col].value_counts(), color='Purple')
            sns.despine(left=True)            
            plt.xticks(rotation=90)
            plt.ylabel('Count')
            plt.title('Distribution of {}'.format(str.capitalize(col)))
        
            plt.subplot(122)
            sns.boxplot(df[col], df[tgt], palette='Set2_r')
            sns.despine(left=True)
            plt.xticks(rotation=90)
            plt.ylabel('%s' % str.capitalize(tgt))
            plt.title('Boxplot of %s to %s' % (str.capitalize(tgt), str.capitalize(col)))
            plt.show()
    
            
            
def plt_feat_classif(df, col, tgt, label=[], title=None, max_col_vals=None):
    
    majority_df = df.loc[df[tgt] == 0]
    minority_df = df.loc[df[tgt] == 1]
    
    if col == tgt:
        pass
    
    elif df[col].dtype == np.float or df[col].dtype == np.integer:
        
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
        box = sns.boxplot(x=df[col], y=df[tgt], orient='h', showmeans=True, ax=axs[0])
        label = label
        sns.kdeplot(majority_df[col], shade=True, label=label[0], ax=axs[1])
        kde = sns.kdeplot(minority_df[col], shade=True, label=label[1], ax=axs[1])
        kde.legend(labels=label, loc=0)
        sns.despine(left=True)
        kde.set(xlabel=col, ylabel='', yticks=([]), title='%s Distribution' % str.capitalize(col))
        box.set(xlabel=col, ylabel=tgt, title='%s Comparison' % str.capitalize(col))
        plt.show()                  
    
    elif df[col].dtype == 'object' or df[col].dtype == 'category':
        
        if len(df[col].unique()) < max_col_vals:
            plt.figure(figsize=(7, 8))
            sns.barplot(x=(round(cat_mean(df, col, tgt), 4) * 100).values, 
                        y=cat_mean(df, col, tgt).index, orient='h')
            sns.despine(left=True)
            plt.xlabel('Percentage (%)')
            plt.ylabel('')
            plt.title('Percentage of %s with respect to %s' % (str.capitalize(title), str.capitalize(col)))
            plt.show()
            
            
            
def missing_df(df, miss_thresh):
    
    """Return a dataframe with a 'count' and 'percentage' column of missing values.
    Arguments:
    df: the dataframe of interest
    miss_thresh: the columns from the df we want to return if miss_thresh greater than 
    some percentage of missing values, ie if set=0.15 - return only columns that are missing more than 15% of the data points
    """
    
    missing = pd.DataFrame()
    count = df.isnull().sum()
    missing['count'] = count[count > 0]
    missing['percentage'] = missing['count'].apply(lambda x: x / df.shape[0])
    missing.sort_values('percentage', ascending=False, inplace=True)
    if missing.empty:
        print('The dataset does not contain any missing values')
    else:
        return missing.loc[missing['percentage'] > miss_thresh]
    
    
    
def heatmap(df, cols, tgt=None, cmap=None):
    '''plot a correlation matrix of specified columns'''
    plt.figure(figsize=(8, 7))
    sns.heatmap(df[cols + [tgt]].corr(), cmap=cmap, annot=True)
    plt.title('Correlation Matrix')
    plt.show()


# In[ ]:




