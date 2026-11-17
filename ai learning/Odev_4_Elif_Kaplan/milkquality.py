# Elif Kaplan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("Milk.csv")

# Verileri manuel olarak sayısal değerlere çevirme -- LabelEncoder hataları aldığım için böyle yaptım  --
data['pH'] = data['pH'].map({'low': 0, 'medium': 1, 'high': 2})
data['Taste'] = data['Taste'].map({'no': 0, 'yes': 1})
data['Odor'] = data['Odor'].map({'no': 0, 'yes': 1})
data['Fat'] = data['Fat'].map({'no': 0, 'yes': 1})
data['Turbidity'] = data['Turbidity'].map({'no': 0, 'yes': 1})

# veri setindeki bazı özelliklerin NaN (eksik) değerler içerdiğini gösteren bir hata aldım 
# NaN değerleri içeren verileri doğrudan Naive Bayes modeline veremeyeceğimiz için verilerdeki NaN değerleri ortalama ile doldurup o şekilde devam ettim
imputer = SimpleImputer(strategy="mean")
data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])  # Sadece özellik sütunlarına uyguluyoruz

X = data[['pH', 'Taste', 'Odor', 'Fat', 'Turbidity']]
y = data['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20, test_size=10, random_state=0)

# Bayes modelini oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = model.predict(X_test)

# Model doğruluğunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")

# Tahmin
unknown_sample = pd.DataFrame({
    'pH': [0],         # low
    'Taste': [1],      # yes
    'Odor': [1],       # yes
    'Fat': [0],        # no
    'Turbidity': [1]   # yes
})

predicted_quality = model.predict(unknown_sample)
print(f"Tahmin Edilen Süt Kalitesi: {predicted_quality[0]}")
