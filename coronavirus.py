"""
Created on Sun Jun 14 17:41:56 2020

@author: deniz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
veriler= pd.read_csv('day_wise.csv')
veriler2=pd.read_csv('full_grouped.csv')

#Görüntüleme
#Dünyada Korona Virüsü Günlük Artış Grafiği
daily_cases=np.array(veriler['New cases'])
daily_cases=daily_cases.reshape(-1,1)
daily_recovered=np.array(veriler['New recovered'])
daily_deaths=np.array(veriler['New deaths'])
plt.scatter(np.arange(0,143),daily_cases,color="blue")
plt.scatter(np.arange(0,143),daily_recovered,color="green")
plt.scatter(np.arange(0,143),daily_deaths,color="red")
plt.xlabel("Korona Virüsü Gün Sayısı")
plt.ylabel("Korona Virüsü Vaka Sayısı")
plt.title("Tüm Dünyada Korona Virüsü Günlük Vaka Artış Sayısı")
plt.show()
#Dünya Üzerindeki Korona Virüsü Aktif Ölüm ve İyileşme Dağılımı
sum_daily_active=veriler['Active']
sum_daily_active=sum_daily_active.iloc[142]
sum_daily_recovered=daily_recovered.sum()
sum_daily_deaths=daily_deaths.sum()
sum_for_daily=[sum_daily_active,sum_daily_recovered,sum_daily_deaths]
sum_for_daily_labels=['Active','Recovered','Deaths']
explode = (0.1, 0, 0)
plt.pie(sum_for_daily,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Dünya Üzerindeki Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

#For Europe
europe=veriler2[veriler2['WHO Region']=="Europe"]
sum_europe_cases=europe['New cases'].sum()
sum_europe_deaths=europe['New deaths'].sum()
sum_europe_recovered=europe['New recovered'].sum()
sum_europe_active=sum_europe_cases-(sum_europe_deaths+sum_europe_recovered)
sum_for_daily_europe=[sum_europe_active,sum_europe_recovered,sum_europe_deaths]
plt.pie(sum_for_daily_europe,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Avrupada Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

plt.bar(np.arange(3),sum_for_daily_europe)
plt.xticks(np.arange(3),('Active', 'Recovered', 'Deaths'))
plt.title("Avrupa Aktif,İyileşen ve Ölüm Sayıları")
plt.show()

#For Africa
africa=veriler2[veriler2['WHO Region']=="Africa"]
sum_africa_cases=africa['New cases'].sum()
sum_africa_deaths=africa['New deaths'].sum()
sum_africa_recovered=africa['New recovered'].sum()
sum_africa_active=sum_africa_cases-(sum_africa_deaths+sum_africa_recovered)
sum_for_daily_africa=[sum_africa_active,sum_africa_recovered,sum_africa_deaths]
plt.pie(sum_for_daily_africa,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Afrikada Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

plt.bar(np.arange(3),sum_for_daily_africa)
plt.xticks(np.arange(3),('Active', 'Recovered', 'Deaths'))
plt.title("Afrika Aktif,İyileşen ve Ölüm Sayıları")
plt.show()
#For Americas
amerika=veriler2[veriler2['WHO Region']=="Americas"]
sum_amerika_cases=amerika['New cases'].sum()
sum_amerika_deaths=amerika['New deaths'].sum()
sum_amerika_recovered=amerika['New recovered'].sum()
sum_amerika_active=sum_amerika_cases-(sum_amerika_deaths+sum_amerika_recovered)
sum_for_daily_amerika=[sum_africa_active,sum_amerika_recovered,sum_amerika_deaths]
plt.pie(sum_for_daily_amerika,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Amerika Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

plt.bar(np.arange(3),sum_for_daily_amerika)
plt.xticks(np.arange(3),('Active', 'Recovered', 'Deaths'))
plt.title("Amerika Aktif,İyileşen ve Ölüm Sayıları")
plt.show()
#Eastern Mediterranean
medit=veriler2[veriler2['WHO Region']=="Eastern Mediterranean"]
sum_medit_cases=medit['New cases'].sum()
sum_medit_deaths=medit['New deaths'].sum()
sum_medit_recovered=medit['New recovered'].sum()
sum_medit_active=sum_medit_cases-(sum_medit_deaths+sum_medit_recovered)
sum_for_daily_medit=[sum_medit_active,sum_medit_recovered,sum_medit_deaths]
plt.pie(sum_for_daily_medit,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Orta Doğu Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

plt.bar(np.arange(3),sum_for_daily_medit)
plt.xticks(np.arange(3),('Active', 'Recovered', 'Deaths'))
plt.title("Orta Asya Aktif,İyileşen ve Ölüm Sayıları")
plt.show()
#South-East Asia
south=veriler2[veriler2['WHO Region']=="South-East Asia"]
sum_south_cases=south['New cases'].sum()
sum_south_deaths=south['New deaths'].sum()
sum_south_recovered=south['New recovered'].sum()
sum_south_active=sum_south_cases-(sum_south_deaths+sum_south_recovered)
sum_for_daily_south=[sum_south_active,sum_south_recovered,sum_south_deaths]
plt.pie(sum_for_daily_south,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Güney Doğu Asya Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

plt.bar(np.arange(3),sum_for_daily_south)
plt.xticks(np.arange(3),('Active', 'Recovered', 'Deaths'))
plt.title("Güney Doğu Asya Aktif,İyileşen ve Ölüm Sayıları")
plt.show()
#Western Pacific
pacific=veriler2[veriler2['WHO Region']=="Western Pacific"]
sum_pacific_cases=pacific['New cases'].sum()
sum_pacific_deaths=pacific['New deaths'].sum()
sum_pacific_recovered=pacific['New recovered'].sum()
sum_pacific_active=sum_pacific_cases-(sum_pacific_deaths+sum_pacific_recovered)
sum_for_daily_pacific=[sum_pacific_active,sum_pacific_recovered,sum_pacific_deaths]
plt.pie(sum_for_daily_pacific,labels=sum_for_daily_labels,autopct='%1.1f%%', shadow=True, explode=explode)
plt.title("Batı Pasific Korona Virüsü Aktif,Ölüm ve İyileşme Dağılımı")
plt.show()

plt.bar(np.arange(3),sum_for_daily_pacific)
plt.xticks(np.arange(3),('Active', 'Recovered', 'Deaths'))
plt.title("Pasifik Aktif,İyileşen ve Ölüm Sayıları")
plt.show()

#Modelleme
turkey=veriler2[veriler2['Country/Region']=="Turkey"]
plt.scatter(np.arange(143),turkey['Confirmed'],color="blue")
plt.scatter(np.arange(143),turkey['Recovered'],color="green")
plt.scatter(np.arange(143),turkey['Deaths'],color="red")
plt.show()
turkey.drop(columns=['Date','Country/Region','WHO Region','New cases','New recovered','New deaths'],inplace=True)

#Normalization
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df = ss.fit_transform(turkey)

#Split Train and Test Parts
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.arange(143),df ,test_size=0.33, random_state=42)
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
#Regression
#Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train[0])
LinearRegressionPrediction=lr.predict(x_test)
plt.scatter(x_test,y_test[0])
plt.plot(x_test,LinearRegressionPrediction)
plt.show()
#Linear Regression Yorumu : Grafikte Görüldüğü gibi veri polinomal bir veri ve doğrusal bir kesme ile doğru tahminler yapamadık.Linear Regression Algoritması Basarısız

from sklearn.metrics import r2_score
print("Linear Regression Algoritmasının Basarı Oranı:",r2_score(y_test[0], LinearRegressionPrediction))#Linear Regression Algoritmasının Basarı Oranı
print(lr.coef_)#Eğimi
print(lr.intercept_)#Başlangıc Yüksekliği

#Polinomal Regression

from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(x_train)

model = LinearRegression()
model.fit(x_poly, y_train[0])
y_poly_pred = model.predict(x_poly)
y_poly_pred=pd.DataFrame(y_poly_pred)
plt.scatter(x_train,y_train[0])
plt.scatter(x_train,y_poly_pred)
#Polinom Regression Yorumu:Grafikde Görüldüğü Üzere Gerçek Değere Çok Daha Yakın Sonuclar Çıktı Buda Polinomal Regressionun Linear Regressiona Göre Çok daha Verimli Olduğu Anlamına Geliyor
































