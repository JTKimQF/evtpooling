import numpy as np
import pandas as pd

from evtpooling import etl_pipeline, alpha_var_agg_plot, alpha_var_sing_plot

losses_week = etl_pipeline(r'C:\Users\jkim595\OneDrive - KPMG\Documents\Thesis\venv\data\STOXX_complete_data.csv', save_file=True)

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

dict_gvkey_conm = {
    12673: "BARCLAYS PLC", 24578: "SWEDBANK AB", 14140: "BANCO SANTANDER SA", 24584: "BANCA MONTE DEI PASCHI SIENA",
    15181: "BANCO BILBAO VIZCAYA ARGENTA", 24826: "TOPDANMARK AS", 15509: "HSBC HLDGS PLC", 28838: "CREDIT SUISSE GROUP",
    15522: "BANCO POPULAR ESPANOL", 62410: "ALPHA SERVICES AND HOLDINGS", 15532: "BNP PARIBAS", 63120: "AXA SA",
    15545: "AVIVA PLC", 63546: "SWISS RE LTD", 15547: "NATIXIS", 63590: "BANK OF IRELAND GROUP PLC", 
    15549: "UNICREDIT SPA", 63759: "SCOR SE", 15552: "DANSKE BANK AS", 144496: "UBS GROUP AG",
    15575: "COMMERZBANK AG", 200646: "BANCA POPOLARE DI SONDRIO", 15576: "DEUTSCHE BANK AG", 200647: "BPER BANCA SPA", 
    15593: "MEDIOBANCA BANCA DI CREDITO", 200705: "DNB BANK ASA", 15598: "AEGON LTD", 204719: "KOMERCNI BANKA AS", 
    15600: "JYSKE BANK", 204825: "HANNOVER RUECK SE", 15603: "LEGAL & GEN GROUP PLC", 205699: "SWISS LIFE HOLDING",
    15617: "ING GROEP NV", 209212: "SYDBANK AS", 15634: "NATWEST GROUP PLC", 214659: "ERSTE GROUP BK AG",
    15645: "STANDARD CHARTERED PLC", 214881: "NORDEA BANK ABP", 15654: "SVENSKA HANDELSBANKEN", 220426: "ZURICH INSURANCE GROUP AG",
    15664: "RSA INSURANCE GROUP PLC", 223504: "ABRDN UK SMALLER COMPANIES", 15671: "SKANDINAVISKA ENSKILDA BANK", 225081: "BANCO BPM SPA",
    15703: "KBC GROUP NV", 225244: "CNP ASSURANCES SA", 15724: "ALLIANZ SE", 227537: "EUROBANK ERGASIAS SERVICES A",
    15773: "SAMPO PLC", 227616: "HISCOX LTD", 15782: "PRUDENTIAL PLC", 229039: "OLD MUTUAL LTD",
    15784: "SOCIETE GENERALE GROUP", 241498: "MPC MUENCHMEYER PETERSEN CAP", 15804: "ASSICURAZIONI GENERALI SPA", 245436: "BANCO DE SABADELL SA",
    15846: "BANKINTER SA", 254578: "BEAZLEY PLC", 15914: "UNIPOLSAI ASSICURAZIONI SPA", 260840: "NN GROUP N.V.",
    15929: "LLOYDS BANKING GROUP PLC", 268159: "DIRECT LINE INSURANCE GRP", 16281: "MAPFRE", 270266: "UNIONE DI BANCHE ITALIANE",
    16348: "INTESA SANPAOLO SPA", 270979: "ADMIRAL GROUP PLC", 16349: "HELVETIA HOLDING AG", 272817: "RAIFFEISEN BANK INTERNATI AG",
    16447: "AGEAS SA/NV", 274282: "TRYG AS", 23087: "BALOISE HOLDING AG", 274774: "LANCASHIRE HLDGS",
    23848: "BANCO COMERCIAL PORTUGUES SA", 286879: "CAIXABANK SA", 23913: "ST JAMES PLACE", 288153: "PHOENIX GROUP HLDGS",
    24563: "CREDIT AGRICOLE SA", 292854: "JULIUS BAER GRUPPE AG", 296091: "GJENSIDIGE FORSIKRING BA"
}

# Convert list of tuples to a list of column keys (MultiIndex format)
valid_columns = pd.MultiIndex.from_tuples(gvkey_curcdd_iid_list, names=losses_week.columns.names)

# Filter the DataFrame to keep only those columns
losses_week_best = losses_week.loc[:, losses_week.columns.isin(valid_columns)]
losses_week_dict = {col: losses_week_best[col].dropna() for col in losses_week_best.columns}

file_path = r'C:\Users\jkim595\OneDrive - KPMG\Documents\Thesis\figures'
alpha_var_agg_plot(losses = losses_week_best,
                   k_values=np.arange(10, 150, 2),
                   vline = 90, threshold = 0.99,
                   dict_gvkey_conm=dict_gvkey_conm, details_on=True)
alpha_var_sing_plot(losses = losses_week_best[(15509, 'GBP', '02W')], 
                    k_values=np.arange(10, 150, 2),
                    file_path=file_path,
                    name = dict_gvkey_conm[15509])