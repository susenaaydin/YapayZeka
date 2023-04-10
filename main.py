import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib


import warnings
warnings.filterwarnings("ignore")


# veri kümesini aktar
data = pd.read_csv('cc_general.csv')
print(data.head())
print(data.info())
data.fillna(method ='ffill', inplace = True)
data.drop('CUST_ID', axis = 1, inplace = True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)


# verilerin birbiri ile ilişkisini anlamak için
plt.style.use('dark_background')


my_colors = ["#ff1493", "#006400", "#800000", "#cd00cd", "#087EB0", "#ff8c00", "#8b658b"]
my_palette = sns.color_palette(my_colors)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14.5, 9))

sns.scatterplot(x='MINIMUM_PAYMENTS', y='PAYMENTS', data=data, ax=axes[0, 0], hue='TENURE', palette=my_palette)
axes[0, 0].set_title('Minimum Payments vs. Payments')

sns.scatterplot(x='CREDIT_LIMIT', y='BALANCE', data=data, ax=axes[0, 1], hue='TENURE', palette=my_palette)
axes[0, 1].set_title('Credit Limit vs. Balance')

sns.scatterplot(x='CREDIT_LIMIT', y='PURCHASES', data=data, ax=axes[0, 2], hue='TENURE', palette=my_palette)
axes[0, 2].set_title('Credit Limit vs. Purchases')

sns.scatterplot(x='PRC_FULL_PAYMENT', y='PURCHASES', data=data, ax=axes[1, 0], hue='TENURE', palette=my_palette)
axes[1, 0].set_title('Full Payment vs. Purchases')

sns.scatterplot(x='MINIMUM_PAYMENTS', y='PRC_FULL_PAYMENT', data=data, ax=axes[1, 1], hue='TENURE', palette=my_palette)
axes[1, 1].set_title('Minimum Payments vs. Full Payments')

sns.scatterplot(x='CASH_ADVANCE', y='PURCHASES', data=data, ax=axes[1, 2], hue='TENURE', palette=my_palette)
axes[1, 2].set_title('Cash Advance vs. Purchases')
plt.show()



#Verilerin birbirleri ile ilişkilerini anlamak için korelasyonlara bakılması (corr)

my_colors2 = ["#ffb5c5", "#ff69b4", "#8b3a62", "#d02090", "#da70d6", "#7a378b", "#ab82ff"]
my_palette2 = sns.color_palette(my_colors2)

fig2 = plt.figure(figsize=(13, 7))
corr_matrix = data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap=my_palette2)
plt.show()



scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# Elbow yöntemi ile küme sayısının belirlenmesi
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

# K-Means algoritması kullanılarak müşterilerin sınıflandırılması
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
pred_y = kmeans.fit_predict(data_scaled)
# PCA ile boyut azaltma gerçekleştirilmesi
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.DataFrame({'cluster': pred_y})], axis = 1)






