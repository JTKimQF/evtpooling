import numpy as np

from evtpooling import (
    alpha_var_agg_plot,
    alpha_var_sing_plot,
    autocorrelation_test,
    basel_backtesting,
    chenzhou,
    etl_pipeline,
    get_alpha_dict,
    get_var_dict,
    kmeans_plot,
    kmeans_pooling,
    loss_return_plot,
    normality_test,
    rmtc_pooling,
    sener_backtesting,
    stationarity_test,
    wald_test,
)

# === File paths ===
file_path = r"C:\Users\jkim595\OneDrive - KPMG\Documents\Thesis\figures"


# === ETL pipeline ===
losses_week, losses_day = etl_pipeline(
    r"C:\Users\jkim595\OneDrive - KPMG\Documents\Thesis\venv\data\STOXX_complete_data.csv",
    return_day_data=True,
    save_file=True,
)


# === Dictionary for gvkey to company name mapping ===
dict_gvkey_conm = {
    12673: "BARCLAYS PLC",
    24578: "SWEDBANK AB",
    14140: "BANCO SANTANDER SA",
    24584: "BANCA MONTE DEI PASCHI SIENA",
    15181: "BANCO BILBAO VIZCAYA ARGENTA",
    24826: "TOPDANMARK AS",
    15509: "HSBC HLDGS PLC",
    28838: "CREDIT SUISSE GROUP",
    15522: "BANCO POPULAR ESPANOL",
    62410: "ALPHA SERVICES AND HOLDINGS",
    15532: "BNP PARIBAS",
    63120: "AXA SA",
    15545: "AVIVA PLC",
    63546: "SWISS RE LTD",
    15547: "NATIXIS",
    63590: "BANK OF IRELAND GROUP PLC",
    15549: "UNICREDIT SPA",
    63759: "SCOR SE",
    15552: "DANSKE BANK AS",
    144496: "UBS GROUP AG",
    15575: "COMMERZBANK AG",
    200646: "BANCA POPOLARE DI SONDRIO",
    15576: "DEUTSCHE BANK AG",
    200647: "BPER BANCA SPA",
    15593: "MEDIOBANCA BANCA DI CREDITO",
    200705: "DNB BANK ASA",
    15598: "AEGON LTD",
    204719: "KOMERCNI BANKA AS",
    15600: "JYSKE BANK",
    204825: "HANNOVER RUECK SE",
    15603: "LEGAL & GEN GROUP PLC",
    205699: "SWISS LIFE HOLDING",
    15617: "ING GROEP NV",
    209212: "SYDBANK AS",
    15634: "NATWEST GROUP PLC",
    214659: "ERSTE GROUP BK AG",
    15645: "STANDARD CHARTERED PLC",
    214881: "NORDEA BANK ABP",
    15654: "SVENSKA HANDELSBANKEN",
    220426: "ZURICH INSURANCE GROUP AG",
    15664: "RSA INSURANCE GROUP PLC",
    223504: "ABRDN UK SMALLER COMPANIES",
    15671: "SKANDINAVISKA ENSKILDA BANK",
    225081: "BANCO BPM SPA",
    15703: "KBC GROUP NV",
    225244: "CNP ASSURANCES SA",
    15724: "ALLIANZ SE",
    227537: "EUROBANK ERGASIAS SERVICES A",
    15773: "SAMPO PLC",
    227616: "HISCOX LTD",
    15782: "PRUDENTIAL PLC",
    229039: "OLD MUTUAL LTD",
    15784: "SOCIETE GENERALE GROUP",
    241498: "MPC MUENCHMEYER PETERSEN CAP",
    15804: "ASSICURAZIONI GENERALI SPA",
    245436: "BANCO DE SABADELL SA",
    15846: "BANKINTER SA",
    254578: "BEAZLEY PLC",
    15914: "UNIPOLSAI ASSICURAZIONI SPA",
    260840: "NN GROUP N.V.",
    15929: "LLOYDS BANKING GROUP PLC",
    268159: "DIRECT LINE INSURANCE GRP",
    16281: "MAPFRE",
    270266: "UNIONE DI BANCHE ITALIANE",
    16348: "INTESA SANPAOLO SPA",
    270979: "ADMIRAL GROUP PLC",
    16349: "HELVETIA HOLDING AG",
    272817: "RAIFFEISEN BANK INTERNATI AG",
    16447: "AGEAS SA/NV",
    274282: "TRYG AS",
    23087: "BALOISE HOLDING AG",
    274774: "LANCASHIRE HLDGS",
    23848: "BANCO COMERCIAL PORTUGUES SA",
    286879: "CAIXABANK SA",
    23913: "ST JAMES PLACE",
    288153: "PHOENIX GROUP HLDGS",
    24563: "CREDIT AGRICOLE SA",
    292854: "JULIUS BAER GRUPPE AG",
    296091: "GJENSIDIGE FORSIKRING BA",
}


# === IID assumption testing ===
normality_test(losses_week, drop=True, dict_gvkey_conm=dict_gvkey_conm)
autocorrelation_test(losses_week, drop=True, dict_gvkey_conm=dict_gvkey_conm)
stationarity_test(losses_week, drop=True, dict_gvkey_conm=dict_gvkey_conm)

normality_test(losses_day, drop=True, dict_gvkey_conm=dict_gvkey_conm)
autocorrelation_test(losses_day, drop=True, dict_gvkey_conm=dict_gvkey_conm)
stationarity_test(losses_day, drop=True, dict_gvkey_conm=dict_gvkey_conm)

losses_week_est = losses_week[losses_week.index <= "2018-12-31"]
losses_day_est = losses_day[losses_day.index <= "2018-12-31"]

losses_week_eval = losses_week[losses_week.index > "2018-12-31"]
losses_day_eval = losses_day[losses_day.index > "2018-12-31"]


# === Losses dictionaries ===
losses_week_dict = {col: losses_week_est[col].dropna() for col in losses_week_est.columns}
losses_day_dict = {col: losses_day_est[col].dropna() for col in losses_day_est.columns}


# === Alpha & VaR Plots ===
k_thresh = 60
alpha_var_agg_plot(
    losses=losses_week_est,
    k_values=np.arange(10, 150, 2),
    vline=k_thresh,
    threshold=0.99,
    dict_gvkey_conm=dict_gvkey_conm,
    details_on=True,
)
alpha_var_agg_plot(
    losses=losses_day_est,
    k_values=np.arange(10, 400, 2),
    vline=250,
    threshold=0.99,
    dict_gvkey_conm=dict_gvkey_conm,
    details_on=True,
)
alpha_var_sing_plot(
    losses=losses_week_est[(63590, "EUR", "01W")],
    k_values=np.arange(10, 100, 2),
    name=dict_gvkey_conm[15509],
)
loss_return_plot(
    losses=losses_week, dict_gvkey_conm=dict_gvkey_conm, plots_per_batch=16, num_columns=4
)


# === Common tail index testing ===
alpha_dict = get_alpha_dict(losses_week_est, k_threshold=k_thresh)
alpha_list = list(alpha_dict.values())
mean_alpha = np.mean(alpha_list)

chenzou_stat, chenzou_p_value = chenzhou(
    alpha_list=alpha_list, common_alpha=mean_alpha, k_threshold=k_thresh
)
wald_stat, wald_p_value = wald_test(
    losses=losses_week_est, alpha_list=alpha_list, common_alpha=mean_alpha, k_threshold=k_thresh
)

print(f"Chenzhou statistic: {chenzou_stat}, p-value: {chenzou_p_value}")
print(f"Wald statistic: {wald_stat}, p-value: {wald_p_value}")


# === KMeans Clustering ===
df_pooling_kmeans = kmeans_pooling(losses=losses_week_est, k_threshold=k_thresh, num_clusters=4)
kmeans_var_dict = {
    index: value
    for index, value in zip(df_pooling_kmeans.index, df_pooling_kmeans["kmeans_var"], strict=False)
}
print(df_pooling_kmeans.iloc[:, :])

kmeans_plot(
    losses=losses_week_est, k_threshold=k_thresh, k_range=np.arange(1, 10), plot_inertia=True
)


# === RMTC Clustering ===
df_pooling_rmtc, beta_list = rmtc_pooling(
    losses=losses_week_est,
    k_threshold=k_thresh,
    beta=([0.4, 0.4, 0.2], [(1.9, 0.2), (2.3, 0.3), (3.0, 0.5)]),
    threshold=1e-4,
    max_iter=50,
)
rmtc_var_dict = {
    index: value
    for index, value in zip(df_pooling_rmtc.index, df_pooling_rmtc["rmtc_var"], strict=False)
}
print("RMC Estimated parameters:\n")
print(df_pooling_rmtc.iloc[:, :], "\n")
print(
    f"Estimated parameters:\n"
    f"π={[round(float(pi), 3) for pi in beta_list[-2][0]]}\n"
    f"ϕ={[(round(float(mu), 3), round(float(sigma), 3)) for mu, sigma in beta_list[-2][1]]}"
)


# === Backtesting ===
losses_week_eval.columns = [col[0] for col in losses_week_eval.columns]

df_backtesting_non_pooling = basel_backtesting(
    losses=losses_week_eval,
    var_dict=get_var_dict(losses_week_est, k_threshold=k_thresh),
    confidence=0.99,
)
df_backtesting_non_pooling.rename(index=dict_gvkey_conm, inplace=True)
print("Backtesting results non-pooling:")
print(df_backtesting_non_pooling.iloc[:, :], "\n")


df_backtesting_kmeans = basel_backtesting(
    losses=losses_week_eval, var_dict=kmeans_var_dict, confidence=0.99
)
df_backtesting_kmeans.rename(index=dict_gvkey_conm, inplace=True)
print("Backtesting results KMeans:")
print(df_backtesting_kmeans.iloc[:, :], "\n")


df_backtesting_rmtc = basel_backtesting(
    losses=losses_week_eval, var_dict=rmtc_var_dict, confidence=0.99
)
df_backtesting_rmtc.rename(index=dict_gvkey_conm, inplace=True)
print("Backtesting results RMTC:")
print(df_backtesting_rmtc.iloc[:, :], "\n")


# === Sener backtesting ===
df_sener_backtesting_non_pooling = sener_backtesting(
    losses=losses_week_eval, var_dict=get_var_dict(losses_week_est, k_threshold=k_thresh), theta=0.5
)
df_sener_backtesting_non_pooling.rename(index=dict_gvkey_conm, inplace=True)
print("Sener backtesting results non-pooling:")
print(df_sener_backtesting_non_pooling.iloc[:, :], "\n")


df_sener_backtesting_kmeans = sener_backtesting(
    losses=losses_week_eval, var_dict=kmeans_var_dict, theta=0.5
)
df_sener_backtesting_kmeans.rename(index=dict_gvkey_conm, inplace=True)
print("Sener backtesting results KMeans:")
print(df_sener_backtesting_kmeans.iloc[:, :], "\n")


df_sener_backtesting_rmtc = sener_backtesting(
    losses=losses_week_eval, var_dict=rmtc_var_dict, theta=0.5
)
df_sener_backtesting_rmtc.rename(index=dict_gvkey_conm, inplace=True)
print("Sener backtesting results RMTC:")
print(df_sener_backtesting_rmtc.iloc[:, :], "\n")
