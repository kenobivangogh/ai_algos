import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Elif Kaplan 

data = pd.read_csv("CNCToolWear2.csv")

fig, axs = plt.subplots(1, 3, figsize=(20, 5)) 

# 1. Parça Besleme Hızı (Feed Rate) ile Aşınma (Tool Wear) Arasındaki İlişki

# Bağımsız ve bağımlı değişkenlerin tanımlanması
X_feed_rate = data[['Feed_Rate']].values
y_tool_wear = data['Tool_Wear'].values

# Model 1
model_feed_rate = LinearRegression()
model_feed_rate.fit(X_feed_rate, y_tool_wear)

# Tahmin edilen aşınma
y_pred_feed_rate = model_feed_rate.predict(X_feed_rate)

# MAE ve R-kare değerleri
mae_feed_rate = mean_absolute_error(y_tool_wear, y_pred_feed_rate)
r2_feed_rate = r2_score(y_tool_wear, y_pred_feed_rate)

axs[0].scatter(X_feed_rate, y_tool_wear, color='black', label='Ölçüm Noktaları')
axs[0].plot(X_feed_rate, y_pred_feed_rate, color='red', label='Regresyon Doğrusu')
axs[0].set_xlabel("Feed Rate (mm/min)")
axs[0].set_ylabel("Tool Wear (mm)")
axs[0].set_title("Feed Rate vs Tool Wear")
axs[0].legend()

# 2. Kesme Derinliği (Depth of Cut) ile Aşınma (Tool Wear) Arasındaki İlişki

# Bağımsız ve bağımlı değişkenler
X_depth_of_cut = data[['Depth_Of_Cut']].values

# Model 2
model_depth_of_cut = LinearRegression()
model_depth_of_cut.fit(X_depth_of_cut, y_tool_wear)

# Tahmin edilen aşınma
y_pred_depth_of_cut = model_depth_of_cut.predict(X_depth_of_cut)

# MAE ve R-kare hesaplamaları
mae_depth_of_cut = mean_absolute_error(y_tool_wear, y_pred_depth_of_cut)
r2_depth_of_cut = r2_score(y_tool_wear, y_pred_depth_of_cut)

axs[1].scatter(X_depth_of_cut, y_tool_wear, color='black', label='Ölçüm Noktaları')
axs[1].plot(X_depth_of_cut, y_pred_depth_of_cut, color='red', label='Regresyon Doğrusu')
axs[1].set_xlabel("Depth of Cut (mm)")
axs[1].set_ylabel("Tool Wear (mm)")
axs[1].set_title("Depth of Cut vs Tool Wear")
axs[1].legend()

# 3. Hız (Speed) ile Aşınma (Tool Wear) Arasındaki İlişki

# Bağımsız ve bağımlı değişkenler
X_speed = data[['Speed']].values

# Model 3
model_speed = LinearRegression()
model_speed.fit(X_speed, y_tool_wear)

# Tahmin edilen aşınma
y_pred_speed = model_speed.predict(X_speed)

# MAE ve R-kare hesaplamaları
mae_speed = mean_absolute_error(y_tool_wear, y_pred_speed)
r2_speed = r2_score(y_tool_wear, y_pred_speed)

axs[2].scatter(X_speed, y_tool_wear, color='black', label='Ölçüm Noktaları')
axs[2].plot(X_speed, y_pred_speed, color='red', label='Regresyon Doğrusu')
axs[2].set_xlabel("Speed (RPM)")
axs[2].set_ylabel("Tool Wear (mm)")
axs[2].set_title("Speed vs Tool Wear")
axs[2].legend()

# PERFORMANS ÇIKTILARININ YORUMLANMASI

results = {
    'Feed Rate': {'MAE': mae_feed_rate, 'R^2': r2_feed_rate},
    'Depth of Cut': {'MAE': mae_depth_of_cut, 'R^2': r2_depth_of_cut},
    'Speed': {'MAE': mae_speed, 'R^2': r2_speed}
}

for param, metrics in results.items():
    print(f"{param} ile Tool Wear ilişkisi:")
    print(f"  MAE: {metrics['MAE']}")
    print(f"  R^2: {metrics['R^2']}")
    print()

best_param = max(results, key=lambda x: results[x]['R^2'])
best_r2 = results[best_param]['R^2']

print(f"En yüksek R^2 değerine sahip parametre: {best_param}")
print(f"En yüksek R^2 değeri: {best_r2}")

# Yorumlar
print("\nYorumlar:")
print("1) Feed Rate ile Tool Wear arasındaki ilişkiyi incelersek, R^2 değeri {}, bu da besleme hızının aşınma üzerindeki etkisinin ne kadar güçlü olduğunu gösterir.".format(r2_feed_rate))
print("2) Depth of Cut ile Tool Wear arasındaki ilişki incelendiğinde, R^2 değeri {}, bu da kesme derinliğinin aşınma üzerindeki etkisinin ne kadar güçlü olduğunu gösterir.".format(r2_depth_of_cut))
print("3) Speed ile Tool Wear arasındaki ilişkiye bakarsak, R^2 değeri {}'dir.".format(r2_speed))

# En büyük R^2 değeri üzerinden yorumlama
if best_param == 'Feed Rate':
    print("Sonuç olarak, besleme hızı (Feed Rate) parametresi, aşınma üzerinde en güçlü etkiye sahiptir.")
elif best_param == 'Depth of Cut':
    print("Sonuç olarak, kesme derinliği (Depth of Cut) parametresi, aşınma üzerinde en güçlü etkiye sahiptir.")
else:
    print("Sonuç olarak, hız (Speed) parametresi, aşınma üzerinde en güçlü etkiye sahiptir.")

# Grafiklerin çizimi

plt.tight_layout()
plt.show()



