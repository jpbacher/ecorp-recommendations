#!/usr/bin/env python
# coding: utf-8

# In[2]:

import glob
import os
import pandas as pd
import numpy as np

def read_dataset(dset, file_type='csv', sep=',', verbose=True):
    
    """ Reads in a dataset.
    
    Arguments:
    dset: a string of the dataset name
    file_type: format of file 
    sep: delimiter to use
    verobse: an option to print out information about the dataset
    
    Returns:
    A pandas dataframe with columns in lowercase, for ease of use later
    """
    if file_type == 'csv':
        df = pd.read_csv('{}.csv'.format(dset), sep=sep)
    elif file_type == 'txt':
        df = pd.read_csv('{}.txt'.format(dset), sep=sep)
    elif file_type == 'excel':
        df = pd.read_csv('{}.xlsx'.format(dset), sep=sep)
    else:
        raise ValueError('Invalid file type: either excel, csv, or txt')
     
    df.columns = df.columns.str.lower()
            
    if verbose:
        print('---------Reading in the dataset: {}.{}---------\n'.format(dset, file_type))
        print('The number instances: %d\n' % df.shape[0])
        print('The number of columns: %d\n' % df.shape[1])
        print('The datatypes of features:\n{}'.format(df.dtypes))

    return df



def read_multiple_dsets(path, format='csv', ignore_index=True, skiprows=None):
    
    if format == 'csv':
        all_files = glob.glob(os.path.join(path, '*.csv'))
        files = (pd.read_csv(f, skiprows=skiprows) for f in all_files)
        df = pd.concat(files, ignore_index=ignore_index)
    elif format == 'excel':
        all_files = glob.glob(os.path.join(path, '*.xlsx'))
        files = (pd.read_xlxs(f, skiprows=skiprows) for f in all_files)
        df = pd.concat(files, ignore_index=ignore_index)
    else:
        raise ValueError('Invalid file format: either excel or csv')
        
    return df



def read_separate_dsets(path, prefix=None, format='csv', sep=',', names=None, skiprows=None, features=None, ignore_index=True, new_path=None, new_name=None, write_to_csv=True):
    
    if format == 'csv':
        all_files = os.listdir(path)
        full_file = []
        for f in all_files:
            if not f.startswith(prefix):
                continue
            data = pd.read_csv(os.path.join(path, f), sep=sep, names=names, skiprows=skiprows)
            data = data[features]
            full_file.append(data)
        
        df = pd.concat(full_file, ignore_index=ignore_index)    
        
    elif format == 'excel':
        all_files = os.listdir(path)
        full_file = []
        for f in all_files:
            if not f.startswith(prefix):
                continue
            data = pd.read_csv(os.path.join(path, f), sep=sep, names=names, skiprows=skiprows)
            data = data[relevant_features[prefix]]
            full_file.append(data)
        
        df = pd.concat(full_file, ignore_index=ignore_index)
    
    else:
        raise ValueError('Invalid file format: either csv, txt, or excel')
        
    if write_to_csv:
        df.to_csv(os.path.join(new_path,'{}.txt'.format(new_name)), sep=sep, header=features, index=False)
    return df



def get_perf_feats(path, file):

    dictionary = dict()
    with open(os.path.join(path, file), 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            loan_id, foreclosure_date = line.split('|')
            loan_id = int(loan_id)
            if loan_id not in dictionary:
                dictionary[loan_id] = {'foreclosure': 0,
                                       'pymnt_count': 0}
            dictionary[loan_id]['pymnt_count'] += 1
            if len(foreclosure_date.strip()) > 0:
                dictionary[loan_id]['foreclosure'] = 1
    return dictionary
                
                

def merge_data(df1, df2, key=None):
    
    """ Merge together 2 different datasets, matching only instances from both datasets.
    
    Arguments:
    df, df1: datasets to merge
    key: a string, the common key in both datasets used to merge together
    
    Returns:
    1 pandas dataframe
    """
    
    df_merged = pd.merge(df1, df2, on=key, how='inner')
    
    return df_merged



def observation(df):
    
    """A quick look at certain characteristics of a dataset.
    
    Argument:
    df: a dataframe of interest
    subset: a string, the columns to look for duplicate entries
    
    Returns:
    Prints various properties of the training dataframe.
    """
    
    
    print('The shape of the data: {}\n'.format(df.shape))
    print('The first 5 rows:\n{}\n'.format(df.head()))
    print('The datatypes:\n{}\n'.format(df.dtypes))
    
    
    
def get_training_data(df, tgt):
    
    train_feats = read_dataset(df, format='csv')
    target = train_feats.pop(tgt)
    
    return train_feats, target





