import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

veriler = pd.read_csv("veriler.csv")
print(veriler)

print(veriler["ulke"].unique()) #ulkelerin benzersizleri
print(veriler.loc[0].unique()) # ilk objenin uniqleri zaten kendi özellikleri - degisen bir sey yok.
print(veriler.iloc[0]) # ilk objeyi döndürüyor, özelliklerini yazdırıyor
print(veriler.iloc[0,0]) # tr döndürdü, ilk elemanin ilk sütundaki(ulke) degeri dondu. CSV iloc -> [object sirasi, sütun sirasi] -> özellik

print(veriler["ulke"].value_counts())



#
# Pandasın get dummies methoduyla One Hot Encoding yapılabilir.
#

one_hot_encoded_data = pd.get_dummies(veriler,columns=["ulke","boy","kilo","yas","cinsiyet"])
print(one_hot_encoded_data) # objenin değerleri hangisiyse o değer true oluyor, TR'de yaşıyorsa tr değeri (ulke_fr , ulke_tr , ulke_us colomnlarından ulke_tr true diğerleri false)






#
# Sci-kit Kütüphanesiyle One Hot Encoding
#




sci_kit_encoded_data = pd.read_csv("veriler.csv")

sci_kit_encoded_data["ulke"] = sci_kit_encoded_data["ulke"].astype('category')
sci_kit_encoded_data["boy"] = sci_kit_encoded_data["boy"].astype('category')
sci_kit_encoded_data["kilo"] = sci_kit_encoded_data["kilo"].astype('category')
sci_kit_encoded_data["yas"] = sci_kit_encoded_data["yas"].astype('category')
sci_kit_encoded_data["cinsiyet"] = sci_kit_encoded_data["cinsiyet"].astype('category')

print(sci_kit_encoded_data["boy"]) 
# çıktı
#Name: boy, dtype: category
#Categories (22, int64): [125, 129, 130, 133, ..., 185, 187, 190, 193] - küçükten büyüğe boylar sıralanmış


sci_kit_encoded_data["ulkeyeni"] = sci_kit_encoded_data["ulke"].cat.codes # return series of codes as well as the index.
sci_kit_encoded_data["boyyeni"] = sci_kit_encoded_data["boy"].cat.codes # return series of codes as well as the index.

# yeni sütun açılıp ismine ulkeyeni dendi, ulke kategorisini

print(sci_kit_encoded_data["ulkeyeni"].values)
# çıktı -> [1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 0 0 0 0 0 0 0]
# orjinal datada (suanda calistigimiz kod uzerinde de) 1'den 9'a kadar ulkeler tr sonra us sonra fr oldugundan bu degerleri 1 2 0 diye gruplandırdı. cat codes işlemi.

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder() # instance obje

# fit the data and transform it - fit_transform , x ve y datasını alıp yeni x datası üretir.
enc_data = pd.DataFrame(enc.fit_transform(sci_kit_encoded_data[["ulkeyeni","boyyeni"]]).toarray())
print(enc_data)

# Orjinal Datamızla Merge işlemi yaparsak da
merged_dataframemiz = sci_kit_encoded_data.join(enc_data)
print(merged_dataframemiz)

# NOT Here we have converted the enc.fit_transform() method to an array because the fit_transform method of OneHotEncoder returns SpiPy sparse matrix so converting to an array first enables us to save space when we have a huge number of categorical variables. 


# DATAFRAME EXPORT
# dataframeadi.to_csv("dosya_adi.csv")
# dataframeadi.to_csv("dosya_adi.csv, index="False", encoding='utf-8') indeksleme yok, encode hatası veriyorsa utf-8 yap.
merged_dataframemiz.to_csv("merged.csv")


