import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas.api.types import is_numeric_dtype

# From: https://github.com/CDC-DNPAO/CDCAnthro/tree/main

cdc__ref__data = pd.read_csv('CDCref_d.csv')


def set_cols_first(df, cols, intersection=True):
    if intersection:
        cols_to_order = [col for col in cols if col in df.columns]
        new_order = cols_to_order + [col for col in df.columns if col not in cols_to_order]
    else:
        new_order = cols + [col for col in df.columns if col not in cols]
    return df[new_order]

def cz_score(var, l, m, s):
    ls = l * s
    invl = 1 / l
    z = (((var / m) ** l) - 1) / ls
    sdp2 = (m * (1 + 2 * ls) ** invl) - m
    sdm2 = m - (m * (1 - 2 * ls) ** invl)
    mz = np.where(var < m, (var - m) / (0.5 * sdm2), (var - m) / (sdp2 * 0.5))
    return z, mz

def cdcanthro(data, 
              age='age_in_months', 
              wt='weight_kg', 
              ht='height_cm', 
              bmi='bmi', 
              all=False):
    
        
    original_data = data.copy()
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    data['seq_'] = range(1, len(data) + 1)

    nms = [col for col in data.columns if col.lower() == 'sex']
    if len(nms) != 1:
        raise ValueError("A child's sex MUST be named 'sex'; this is case insensitive.")
    if nms[0] != 'sex':
        data.rename(columns={nms[0]: 'sex'}, inplace=True)

    data['age'] = data[age]
    data['wt'] = data[wt]
    data['ht'] = data[ht]

    if 'bmi' in data.columns:
        data['bmi'] = data[bmi]
    else:
        data['bmi'] = data['wt'] / (data['ht'] / 100) ** 2

    if 'age' not in data.columns:
        raise ValueError('There must be a variable for age in months in the data')
    
    
    
    # Create a new column 'sexn' by converting the first character of 'sex' to uppercase
    data['sexn'] = data['sex'].str[0].str.upper()

    # Recode 'sexn' based on the conditions
    data['sexn'] = np.select(
        [
            data['sexn'].isin([1, 'B', 'M']),
            data['sexn'].isin([2, 'G', 'F'])
        ],
        [1, 2],
        default=np.nan  # or a default value if needed
    )

    # Filter the data based on the given conditions
    data = data[(data['age'].between(24, 239.9999)) & ~((data['wt'].isna()) & (data['ht'].isna()))]

    # Select the desired columns
    data = data[['seq_', 'sexn', 'age', 'wt', 'ht', 'bmi']]
        
    
    # Filter the DataFrame based on the conditions
    cdc_ref = cdc__ref__data[(cdc__ref__data['_AGEMOS1'] > 23) & (cdc__ref__data['denom'] == 'age')]
    
    
    cdc_ref.columns = cdc_ref.columns.str.lower()
    cdc_ref.columns = cdc_ref.columns.str.replace('^_', '', regex=True)
    cdc_ref.rename(columns={'sex': 'sexn'}, inplace=True)
    

    # Filter the DataFrame where 'agemos2' equals 240 and select specific columns
    d20 = cdc_ref[cdc_ref['agemos2'] == 240][['sexn', 'agemos2', 'lwt2', 'mwt2', 'swt2', 'lbmi2', 'mbmi2', 'sbmi2', 'lht2', 'mht2', 'sht2']]

    # Rename the columns by removing the '2' from the column names
    d20.columns = d20.columns.str.replace('2', '', regex=True)

    # Select specific columns from the original 'cdc_ref' DataFrame
    cdc_ref = cdc_ref[['sexn', 'agemos1', 'lwt1', 'mwt1', 'swt1', 'lbmi1', 'mbmi1', 'sbmi1', 'lht1', 'mht1', 'sht1']]

    # Rename the columns by removing the '1' from the column names
    cdc_ref.columns = cdc_ref.columns.str.replace('1', '', regex=True)
            
    # Combine 'cdc_ref' and 'd20' DataFrames by appending them
    cdc_ref = pd.concat([cdc_ref, d20], ignore_index=True)

    # Add 'mref' and 'sref' columns based on the condition for 'sexn'
    cdc_ref.loc[cdc_ref['sexn'] == 1, ['mref', 'sref']] = [23.02029, 0.13454]  # checked on 7/9/22
    cdc_ref.loc[cdc_ref['sexn'] == 2, ['mref', 'sref']] = [21.71700, 0.15297]        
    
    
    
        # Create a list of column names
    v = ['sexn', 'age', 'wl', 'wm', 'ws', 'bl', 'bm', 'bs', 'hl', 'hm', 'hs', 'mref', 'sref']

    # Rename the columns of 'cdc_ref' using the list 'v'
    cdc_ref.columns = v
        
    from scipy.interpolate import interp1d
    
    # Get unique ages from the 'data' DataFrame
    uages = data['age'].unique()

    # Calculate the length of the set difference between ages in 'data' and 'cdc_ref'
    dlen = len(set(data['age']) - set(cdc_ref['age']))
    db = cdc_ref[cdc_ref['sexn'] == 1]    # Filter 'cdc_ref' DataFrame for sexn == 1 (boys)

    # Function to interpolate reference data for boys
    def fboys(v):
        f = interp1d(db['age'], v, fill_value="extrapolate")
        return f(uages)

    # Filter 'cdc_ref' DataFrame for sexn == 2 (girls)
    dg = cdc_ref[cdc_ref['sexn'] == 2]

    # Function to interpolate reference data for girls
    def fgirls(v):
        f = interp1d(dg['age'], v, fill_value="extrapolate")
        return f(uages)
    
    
    if dlen > 0:
        if len(uages) > 1:
            # Apply fboys and fgirls to each column in db and dg, respectively, for multiple ages
            db = pd.DataFrame({col: fboys(db[col]) for col in v})
            dg = pd.DataFrame({col: fgirls(dg[col]) for col in v})
        else:
            if len(uages) == 1:  # dataset has only 1 age
                # Apply fboys and fgirls to each column in db and dg, respectively, and transpose the result
                db = pd.DataFrame({col: fboys(db[col]) for col in v}).T
                dg = pd.DataFrame({col: fgirls(dg[col]) for col in v}).T
    
    
    
    # Combine the 'db' and 'dg' DataFrames
    cdc_ref = pd.concat([db, dg], ignore_index=True)

    # Get unique combinations of 'sexn' and 'age' from 'data'
    du = data[['sexn', 'age']].drop_duplicates()

    # Perform a merge to match 'cdc_ref' with 'du' on 'sexn' and 'age'
    cdc_ref = pd.merge(du, cdc_ref, on=['sexn', 'age'], how='left')

    # Ensure 'data' and 'cdc_ref' are sorted by 'sexn' and 'age'
    data = data.sort_values(by=['sexn', 'age'])
    cdc_ref = cdc_ref.sort_values(by=['sexn', 'age'])

    # Merge 'cdc_ref' with 'data'
    dt = pd.merge(data, cdc_ref, on=['sexn', 'age'], how='left')

    # Assuming cz_score is a defined function
    dt['waz'], dt['mod_waz'] = cz_score(dt['wt'], dt['wl'], dt['wm'], dt['ws'])
    dt['haz'], dt['mod_haz'] = cz_score(dt['ht'], dt['hl'], dt['hm'], dt['hs'])
    dt['bz'], dt['mod_bmiz'] = cz_score(dt['bmi'], dt['bl'], dt['bm'], dt['bs'])    
    

    from scipy.stats import norm

    # Rename columns in 'dt' DataFrame
    dt = dt.rename(columns={'bl': 'bmi_l', 'bm': 'bmi_m', 'bs': 'bmi_s'})

    # Drop specified columns
    dt = dt.drop(columns=['wl', 'wm', 'ws', 'hl', 'hm', 'hs'])

    # Add new columns based on the calculations
    dt['bmip'] = 100 * norm.cdf(dt['bz'])
    dt['p50'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.50)) ** (1 / dt['bmi_l'])
    dt['p85'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.85)) ** (1 / dt['bmi_l'])
    dt['p95'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.95)) ** (1 / dt['bmi_l'])
    dt['p97'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.97)) ** (1 / dt['bmi_l'])
    dt['wap'] = 100 * norm.cdf(dt['waz'])
    dt['hap'] = 100 * norm.cdf(dt['haz'])

    # Other BMI metrics
    dt['z1'] = ((dt['bmi'] / dt['bmi_m']) - 1) / dt['bmi_s']
    dt['z0'] = np.log(dt['bmi'] / dt['bmi_m']) / dt['bmi_s']

    # Add more columns based on the calculations
    dt['dist_median'] = dt['z1'] * dt['bmi_m'] * dt['bmi_s']
    dt['adj_dist_median'] = dt['z1'] * dt['sref'] * dt['mref']
    dt['perc_median'] = dt['z1'] * 100 * dt['bmi_s']
    dt['adj_perc_median'] = dt['z1'] * 100 * dt['sref']
    dt['log_perc_median'] = dt['z0'] * 100 * dt['bmi_s']
    dt['adj_log_perc_median'] = dt['z0'] * 100 * dt['sref']
    dt['bmip95'] = 100 * (dt['bmi'] / dt['p95'])
        
    # Add new columns 'ebz', 'ebp', and 'agey'
    dt['ebz'] = dt['bz']
    dt['ebp'] = dt['bmip']
    dt['agey'] = dt['age'] / 12

    # Calculate 'sigma' based on the condition for 'sexn'
    dt['sigma'] = np.where(
        dt['sexn'] == 1,
        0.3728 + 0.5196 * dt['agey'] - 0.0091 * dt['agey'] ** 2,
        0.8334 + 0.3712 * dt['agey'] - 0.0011 * dt['agey'] ** 2
    )

    # Update 'ebp' where 'bmip' is greater than or equal to 95
    dt.loc[dt['bmip'] >= 95, 'ebp'] = 90 + 10 * norm.cdf((dt['bmi'] - dt['p95']) / np.round(dt['sigma'], 8))

    # Update 'ebz' where 'bmip' is greater than or equal to 95 and 'ebp' is less than 100
    dt.loc[(dt['bmip'] >= 95) & (dt['ebp'] / 100 < 1), 'ebz'] = norm.ppf(dt['ebp'] / 100)

    # Set 'ebz' to 8.21 where 'ebp' is exactly 100
    dt.loc[dt['ebp'] / 100 == 1, 'ebz'] = 8.21  # highest possible value is 8.20945

    # Drop the specified columns
    columns_to_drop = ['agey', 'mref', 'sref', 'sexn', 'wt', 'ht']
    dt = dt.drop(columns=columns_to_drop)

    # Rename columns
    dt = dt.rename(columns={
        'bz': 'original_bmiz',
        'bmip': 'original_bmip',
        'ebp': 'bmip',
        'ebz': 'bmiz'
    })
    

    # Define the list of columns 'v'
    v = [
        'seq_', 'bmi', 'bmiz', 'bmip', 'waz', 'wap', 'haz', 'hap', 'p50',
        'p95', 'bmip95', 'original_bmip', 'original_bmiz', 'perc_median',
        'mod_bmiz', 'mod_waz', 'mod_haz'
    ]

    # Add additional columns if 'all' is True
    if all:
        v.extend([
            'bmi_l', 'bmi_m', 'bmi_s', 'sigma', 'adj_dist_median', 'dist_median',
            'adj_perc_median', 'log_perc_median', 'adj_log_perc_median'
        ])

    # Select the columns from 'dt' based on the list 'v'
    dt = dt[v]

    # Drop the 'bmi' column if it exists in 'original_data'
    if 'bmi' in original_data.columns:
        dt = dt.drop(columns=['bmi'])

    # Set 'seq_' as the index and join with 'original_data'
    dt = dt.set_index('seq_')
    original_data = original_data.set_index('seq_')
    dtot = dt.join(original_data, how='right')

    # Reorder columns: put the original columns first
    dtot = dtot[original_data.columns.tolist() + [col for col in dtot.columns if col not in original_data.columns]]

    # Remove the 'seq_' column
    dtot = dtot.reset_index(drop=True)

    # Return the final DataFrame
    return dtot
    

if __name__ == '__main__':
    data = pd.DataFrame({
            'seq_': [1, 2, 3, 4],
            'sex': ['girl', 'girl', 'boy', 'boy'],
            'age_in_months': [108, 110, 110, 110],
            'weight_kg': [15, 12, 14, 16],
            'height_cm': [80, 80, 70, 65],
            'bmi': [15, 18, 17, 18]
        })
    
    res = cdcanthro(data)
    print(res)
    # print(cdc__ref__data)