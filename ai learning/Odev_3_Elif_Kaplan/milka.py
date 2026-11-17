import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Elif Kaplan

data = pd.read_csv('Milk.csv')

# Verileri manuel olarak sayısal değerlere çevirme -- LabelEncoder hataları aldığım için böyle yaptım  --
data['pH'] = data['pH'].map({'low': 0, 'medium': 1, 'high': 2})
data['Taste'] = data['Taste'].map({'no': 0, 'yes': 1})
data['Odor'] = data['Odor'].map({'no': 0, 'yes': 1})
data['Fat'] = data['Fat'].map({'no': 0, 'yes': 1})
data['Turbidity'] = data['Turbidity'].map({'no': 0, 'yes': 1})

X = data[['pH', 'Taste', 'Odor', 'Fat', 'Turbidity']]  # Bağımsız değişkenler
y = data['Quality']  # Bağımlı değişken

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=321)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrisi:")
print(conf_matrix)

# Yeni veriyle tahmin yapma
new_data = pd.DataFrame({
    'pH': [0],  # low -> 0
    'Taste': [1],  # yes -> 1
    'Odor': [1],  # yes -> 1
    'Fat': [0],  # no -> 0
    'Turbidity': [1]  # yes -> 1
})

predicted_quality = model.predict(new_data)
print(f"Süt Kalitesi Tahmini: {predicted_quality[0]}")

plt.figure(figsize=(10,10))
plot_tree(model, filled=True)
plt.show()
