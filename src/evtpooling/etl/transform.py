from fuzzywuzzy import process
import pandas as pd

def auto_detect_categorical(df, threshold=0.05):
    categorical_cols = []

    for col in df.columns:
        n_unique = df[col].nunique()
        n_total = len(df[col])
        if (df[col].dtype == 'object') and (n_unique / n_total < threshold):
            categorical_cols.append(col)

    return categorical_cols

def comparing_categories(dirty_cat_list, clean_cat_list, threshold = 80):

    for cat_clean in clean_cat_list:
        matches = process.extract(cat_clean, dirty_cat_list, limit= dirty_cat_list.shape[0])
        
        for potential_matches in matches:
            if potential_matches[1] >= threshold:
                dirty_cat_list.loc[dirty_cat_list == potential_matches[0]] = cat_clean



