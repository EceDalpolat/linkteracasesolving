import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.utils.logger import get_logger
logger = get_logger()
import openpyxl


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
RAW_DATA_PATH="data/raw/Veri-seti.xlsx"
PROCESSED_DATA_PATH="data/processed/"
plt.style.use('seaborn-v0_8')  
plt.rcParams['figure.figsize'] = (12, 8)  

logger.info(f"Veri yükleniyor: {RAW_DATA_PATH}")
df_ = pd.read_excel(RAW_DATA_PATH)  
df = df_.copy()
logger.info(f"Orijinal boyut: {df.shape}")




def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
check_df(df[num_cols])
# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)



def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True, col_name
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df.head()
for col in num_cols:
    result = check_outlier(df, col)
    print(f'{col} sütunu için sonuç: {result}')


# Eksik gözlem sayısını bulalım
missing_values = df.isnull().sum()
logger.info(f"Veri setindeki eksik değer sayısı:\n{missing_values}")
# Eksik gözlem oranını hesaplayalım
missing_ratio = (df.isnull().sum() / len(df)) * 100

# Sadece eksik değeri olan sütunları filtrele
missing_values_percent = missing_ratio[missing_ratio > 0]

# Eksik değer oranlarına göre sıralama
missing_values_percent = missing_values_percent.sort_values(ascending=False)

# Eksik değer oranlarını yazdır
print(missing_values_percent)

# Eksik değerleri hem sayısal hem de oransal olarak bir tablo halinde görelim
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Missing Ratio (%)': missing_ratio})

# Eksik değeri olan sütunları görelim
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)
df.head()

## Eksik değer oranı %80 olanları None ile geriye kalanı None ile doldurdum .
# 1. %80'den fazla eksik değeri olan kategorik değişkenleri bulalım
missing_values_percent = (df.isnull().sum() / len(df)) * 100
high_missing_columns = missing_values_percent[missing_values_percent > 80].index

# Bu değişkenleri "None" ile dolduralım
df[high_missing_columns] = df[high_missing_columns].fillna('None')

# 2. Geriye kalan eksik değerleri ortalama ile dolduralım (sadece sayısal sütunlar için)
# Sayısal değişkenleri seçip eksik olanları ortalama ile doldurma
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean())

# İşlem sonrası eksik verileri kontrol edelim
print(df.isnull().sum())
logger.info("Eksik değerler dolduruldu ✅")
# Kategorik değişkenlerdeki kategorilerin dağılımını görmek için
for col in cat_cols:
    print(f"Distribution of {col}:")
    print(df[col].value_counts(normalize=True) * 100)
    print("\n")

# Rare encoder fonksiyonu
def rare_encoder(dataframe, rare_perc=0.10):
    """
    Nadir kategorileri belirli bir eşik değerin altındaki frekansa göre "Rare" ile birleştirir.

    Parameters:
    dataframe: DataFrame
    rare_perc: float, nadir kategoriler için frekans eşiği (örneğin, 0.01 = %1)

    Returns:
    dataframe: Nadir kategorilerle birleştirilmiş veri seti
    """
    # Kategorik değişkenlerde nadir kategorileri bulma ve Rare ile değiştirme
    for col in cat_cols:
        # Kategorilerin oranını hesapla
        temp_series = dataframe[col].value_counts() / len(dataframe)

        # Nadir kategorileri belirle (rare_perc'in altında olan kategoriler)
        rare_labels = temp_series[temp_series < rare_perc].index

        # Nadir kategorileri "Rare" ile birleştir
        dataframe[col] = dataframe[col].apply(lambda x: 'Rare' if x in rare_labels else x)

    return dataframe

check_df(df)
logger.info("Aykırı değer kontrolü yapılıyor...")
for col_name in num_cols:
    result = check_outlier(df, col_name)
# logger.info("Aykırı değer kontrolü tamamlandı ✅")
# logger.info("Aykırı değerler baskılanıyor...")
# for col_name in num_cols:
#     replace_with_thresholds(df, col_name)
# logger.info("Aykırı değer baskılama tamamlandı ✅")

# os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
# processed_file = os.path.join(PROCESSED_DATA_PATH, "music_processed.csv")
# df.to_csv(processed_file, index=False)
# logger.info(f"İşlenmiş veri kaydedildi: {processed_file}")