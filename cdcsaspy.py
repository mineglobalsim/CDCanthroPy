import numpy as np
import pandas as pd
from scipy.stats import norm

def set_cols_first(df, cols, intersection=True):
    if intersection:
        cols = [col for col in cols if col in df.columns]
    new_cols = cols + [col for col in df.columns if col not in cols]
    return df[new_cols]

def cz_score(var, l, m, s):
    ls = l * s
    invl = 1 / l
    z = (((var / m) ** l) - 1) / ls
    sdp2 = (m * (1 + 2 * ls) ** invl) - m
    sdm2 = m - (m * (1 - 2 * ls) ** invl)
    mz = np.where(var < m, (var - m) / (0.5 * sdm2), (var - m) / (sdp2 * 0.5))
    return z, mz

def cdcanthro(data, age='age_in_months', wt='weight_kg', ht='height_cm', bmi='bmi', all=False):
    data = data.copy()
    data['seq_'] = np.arange(len(data))

    data['sex'] = data['sex'].str.upper().str[0].replace({'B': 1, 'M': 1, 'G': 2, 'F': 2})
    data['age'] = data[age]
    data['wt'] = data[wt]
    data['ht'] = data[ht]

    if bmi in data.columns:
        data['bmi'] = data[bmi]
    else:
        data['bmi'] = data['wt'] / (data['ht'] / 100) ** 2

    if 'age' not in data.columns:
        raise ValueError('There must be a variable for age in months in the data')

    data = data[(data['age'].between(24, 239.9999)) & ~(data['wt'].isna() & data['ht'].isna())]

    # Assuming `cdc_ref_data` is loaded somewhere before this function is called.
    cdc_ref = cdc_ref_data[(cdc_ref_data['_AGEMOS1'] > 23) & (cdc_ref_data['denom'] == 'age')]

    cdc_ref.columns = cdc_ref.columns.str.lower().str.replace('^_', '', regex=True)
    cdc_ref.rename(columns={'sex': 'sexn'}, inplace=True)

    d20 = cdc_ref[cdc_ref['agemos2'] == 240].copy()
    d20.columns = d20.columns.str.replace('2', '')

    cdc_ref = cdc_ref[['sexn', 'agemos1', 'lwt1', 'mwt1', 'swt1', 'lbmi1', 'mbmi1', 'sbmi1', 'lht1', 'mht1', 'sht1']].copy()
    cdc_ref.columns = cdc_ref.columns.str.replace('1', '')

    cdc_ref = pd.concat([cdc_ref, d20], ignore_index=True)
    cdc_ref.loc[cdc_ref['sexn'] == 1, ['mref', 'sref']] = [23.02029, 0.13454]
    cdc_ref.loc[cdc_ref['sexn'] == 2, ['mref', 'sref']] = [21.71700, 0.15297]

    v = ['sexn', 'age', 'wl', 'wm', 'ws', 'bl', 'bm', 'bs', 'hl', 'hm', 'hs', 'mref', 'sref']

    uages = data['age'].unique()
    dlen = len(set(data['age']) - set(cdc_ref['age']))

    if dlen > 0:
        def approx_fcn(group, uages):
            return pd.DataFrame({col: np.interp(uages, group['age'], group[col]) for col in v})

        db = cdc_ref[cdc_ref['sexn'] == 1].groupby('sexn').apply(approx_fcn, uages).reset_index(drop=True)
        dg = cdc_ref[cdc_ref['sexn'] == 2].groupby('sexn').apply(approx_fcn, uages).reset_index(drop=True)
        cdc_ref = pd.concat([db, dg], ignore_index=True)

    du = data[['sexn', 'age']].drop_duplicates()
    cdc_ref = cdc_ref.merge(du, on=['sexn', 'age'], how='right')

    dt = data.merge(cdc_ref, on=['sexn', 'age'], how='left')

    dt['waz'], dt['mod_waz'] = cz_score(dt['wt'], dt['wl'], dt['wm'], dt['ws'])
    dt['haz'], dt['mod_haz'] = cz_score(dt['ht'], dt['hl'], dt['hm'], dt['hs'])
    dt['bz'], dt['mod_bmiz'] = cz_score(dt['bmi'], dt['bl'], dt['bm'], dt['bs'])

    dt['bmip'] = 100 * norm.cdf(dt['bz'])
    dt['p50'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.50)) ** (1 / dt['bmi_l'])
    dt['p85'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.85)) ** (1 / dt['bmi_l'])
    dt['p95'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.95)) ** (1 / dt['bmi_l'])
    dt['p97'] = dt['bmi_m'] * (1 + dt['bmi_l'] * dt['bmi_s'] * norm.ppf(0.97)) ** (1 / dt['bmi_l'])
    dt['wap'] = 100 * norm.cdf(dt['waz'])
    dt['hap'] = 100 * norm.cdf(dt['haz'])

    dt['z1'] = (dt['bmi'] / dt['bmi_m'] - 1) / dt['bmi_s']
    dt['z0'] = np.log(dt['bmi'] / dt['bmi_m']) / dt['bmi_s']

    dt['dist_median'] = dt['z1'] * dt['bmi_m'] * dt['bmi_s']
    dt['adj_dist_median'] = dt['z1'] * dt['sref'] * dt['mref']
    dt['perc_median'] = dt['z1'] * 100 * dt['bmi_s']
    dt['adj_perc_median'] = dt['z1'] * 100 * dt['sref']
    dt['log_perc_median'] = dt['z0'] * 100 * dt['bmi_s']
    dt['adj_log_perc_median'] = dt['z0'] * 100 * dt['sref']
    dt['bmip95'] = 100 * (dt['bmi'] / dt['p95'])

    dt['ebz'] = dt['bz']
    dt['ebp'] = dt['bmip']
    dt['agey'] = dt['age'] / 12

    dt['sigma'] = np.where(
        dt['sexn'] == 1,
        0.3728 + 0.5196 * dt['agey'] - 0.0091 * dt['agey'] ** 2,
        0.8334 + 0.3712 * dt['agey'] - 0.0011 * dt['agey'] ** 2
    )

    dt.loc[dt['bmip'] >= 95, 'ebp'] = 90 + 10 * norm.cdf((dt['bmi'] - dt['p95']) / np.round(dt['sigma'], 8))
    dt.loc[(dt['bmip'] >= 95) & (dt['ebp'] / 100 < 1), 'ebz'] = norm.ppf(dt['ebp'] / 100)
    dt.loc[dt['ebp'] / 100 == 1, 'ebz'] = 8.21

    dt.drop(columns=['agey', 'mref', 'sref', 'sexn', 'wt', 'ht'], inplace=True)

    dt.rename(columns={
        'bz': 'original_bmiz',
        'bmip': 'original_bmip',
        'ebp': 'bmip',
        'ebz': 'bmiz'
    }, inplace=True)

    v = ['seq_', 'bmi', 'bmiz', 'bmip', 'waz', 'wap', 'haz', 'hap', 'p50',
         'p95', 'bmip95', 'original_bmip', 'original_bmiz', 'perc_median',
         'mod_bmiz', 'mod_waz', 'mod_haz']

    if all:
        v.extend(['bmi_l', 'bmi_m', 'bmi_s', 'sigma', 'adj_dist_median', 'dist_median',
                  'adj_perc_median', 'log_perc_median', 'adj_log_perc_median'])

    dt = dt[v]

    if 'bmi' in data.columns:
        dt.drop(columns=['bmi'], inplace=True)

    dt = set_cols_first(dt, data.columns)
    dt = dt.merge(data, on='seq_', how='right')
    dt.drop(columns=['seq_'], inplace=True)

    return dt
