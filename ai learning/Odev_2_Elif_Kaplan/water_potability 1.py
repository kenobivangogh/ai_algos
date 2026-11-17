import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Elif Kaplan

data = pd.read_csv('water_potability.csv')

# Bağımsız değişken (X) ve bağımlı değişken (y)
X = data.drop('Potability', axis=1)
y = data['Potability']

# (%70 eğitim, %30 test) şeklinde verinin eğitim ve test verileri olarak ayrıştırılmassı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verileri normalize etmek
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Farklı k değerleri için k-NN modelleri
k_values = [3, 5, 7]
best_accuracy = 0
best_k = 0
best_model = None

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'k={k}, Doğruluk: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_model = knn

print(f'\nEn iyi model k={best_k} ile oluşturuldu, Doğruluk: {best_accuracy}')

# Karmaşıklık matrisi
y_pred = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nKarmaşıklık Matrisi:')
print(conf_matrix)

# Tip-I ve Tip-II hata oranları
FP = conf_matrix[0,1] # False positive
FN = conf_matrix[1,0] # False negative
TP = conf_matrix[1,1] # True positive
TN = conf_matrix[0,0] # True negative

tip1_error = FP / (FP + TN)
tip2_error = FN / (FN + TP)

print(f'\nTip-I Hata Oranı: {tip1_error}')
print(f'Tip-II Hata Oranı: {tip2_error}')

print('\nSınıflandırma Raporu:')
print(classification_report(y_test, y_pred))

# Yeni veri ile sınıflandırma
new_data = np.array([[6.51, 211.05, 27193.32, 7.72, 274.45, 442.54, 15.78, 63.51, 3.67]])
new_data_df = pd.DataFrame(new_data, columns=X.columns)
new_data_scaled = scaler.transform(new_data_df)
result = best_model.predict(new_data_scaled)

if result == 1:
    print('\nBu su içilebilir.')
else:
    print('\nBu su içilemez.')

# Tip-I ve Tip-II hata oranlarına göre yorumlar
def create_com(tip1_error, tip2_error):
    com = ""

    # Tip-I hatası yorumu (False Positive)
    if tip1_error < 0.1:
        com += "Tip-I hata oranı oldukça düşük (%10'dan az). Gereksiz alarm sayısı az olacaktır.\n"
    elif tip1_error < 0.3:
        com += "Tip-I hata oranı makul düzeyde (%10-%30 arası). Bazen gereksiz alarmlar olabilir, ancak yönetilebilir.\n"
    else:
        com += "Tip-I hata oranı yüksek. İçilebilir suyu sık sık içilemez olarak sınıflandırabilir. Bu, gereksiz müdahalelere neden olabilir.\n"

    # Tip-II hatası yorumu (False Negative)
    if tip2_error < 0.1:
        com += "Tip-II hata oranı oldukça düşük (%10'dan az). İçilemez suyun yanlış sınıflandırılması riski düşüktür.\n"
    elif tip2_error < 0.3:
        com += "Tip-II hata oranı makul düzeyde (%10-%30 arası). İçilemez suyun bazen içilebilir olarak sınıflandırılması mümkündür, ancak risk kontrol edilebilir.\n"
    else:
        com += "Tip-II hata oranı çok yüksek. İçilemez suyun içilebilir olarak sınıflandırılması sağlık açısından ciddi riskler oluşturabilir.\n"

    return com


com = create_com(tip1_error, tip2_error)
print("Hata Oranları Yorumları:")
print(com)
