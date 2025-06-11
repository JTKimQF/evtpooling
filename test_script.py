import pandas as pd

from evtpooling import etl_pipeline

losses_week = etl_pipeline(r'C:\Users\jkim595\OneDrive - KPMG\Documents\Thesis\evtpooling\data\STOXX_complete_data.csv', save_file=True)

gvkey_curcdd_iid_list = [
    (12673, 'EUR', '03W'), (14140, 'EUR', '01W'), (15181, 'EUR', '03W'), (15509, 'GBP', '02W'), (15545, 'GBP', '02W'),
    (15549, 'EUR', '01W'), (15552, 'EUR', '07W'), (15575, 'EUR', '01W'), (15576, 'EUR', '10W'), (15593, 'EUR', '04W'),
    (15598, 'EUR', '03W'), (15600, 'DKK', '01W'), (15603, 'GBP', '01W'), (15617, 'EUR', '14W'), (15634, 'GBP', '02W'),
    (15645, 'EUR', '10W'), (15654, 'EUR', '07W'), (15664, 'GBP', '02W'), (15703, 'EUR', '01W'), (15724, 'EUR', '01W'),
    (15773, 'EUR', '01W'), (15782, 'GBP', '01W'), (15784, 'EUR', '01W'), (15804, 'EUR', '01W'), (15846, 'EUR', '05W'),
    (15914, 'EUR', '07W'), (16281, 'EUR', '01W'), (16348, 'EUR', '09W'), (16349, 'CHF', '02W'), (16447, 'EUR', '01W'),
    (23087, 'CHF', '01W'), (23848, 'EUR', '07W'), (23913, 'GBP', '01W'), (24563, 'EUR', '02W'), (24578, 'SEK', '01W'),
    (24584, 'EUR', '01W'), (24826, 'DKK', '01W'), (270979, 'GBP', '01W'), (272817, 'EUR', '01W'), (274282, 'DKK', '01W'),
    (274774, 'GBP', '01W'), (286879, 'EUR', '01W'), (62410, 'EUR', '01W'), (63120, 'EUR', '15W'), (63546, 'CHF', '08W'),
    (63590, 'EUR', '02W'), (63759, 'EUR', '07W'), (200647, 'EUR', '01W'), (204825, 'EUR', '01W'), (205699, 'CHF', '04W'),
    (209212, 'DKK', '01W'), (214659, 'EUR', '02W'), (223504, 'GBP', '01W'), (225081, 'EUR', '01W'), (227537, 'EUR', '02W'),
    (227616, 'GBP', '01W'), (229039, 'GBP', '01W'), (241498, 'EUR', '01W'), (245436, 'EUR', '01W')
]

# Convert list of tuples to a list of column keys (MultiIndex format)
valid_columns = pd.MultiIndex.from_tuples(gvkey_curcdd_iid_list, names=losses_week.columns.names)

# Filter the DataFrame to keep only those columns
losses_week_best = losses_week.loc[:, losses_week.columns.isin(valid_columns)]
losses_week_dict = {col: losses_week_best[col].dropna() for col in losses_week_best.columns}

print(losses_week.head())
print(losses_week_dict)