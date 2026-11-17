import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("SimpleDiabets.csv")

X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_accuracy = 0
best_params = {}

# Farklı katman ve nöron kombinasyonları
hidden_layer_configs = [
    (10,), (20,), (10, 10), (20, 10), (30, 30), (50,), (50, 20), (100,), (50, 50, 20)
]
activation_functions = ['logistic', 'tanh', 'relu']

for hidden_layers in hidden_layer_configs:
    for activation in activation_functions:
        # model
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=1000, random_state=42)
        
        mlp.fit(X_train, y_train)
        
        # Test veri seti üzerinde doğruluğu hesapla
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # En iyi doğruluğu ve parametreler update
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'hidden_layer_sizes': hidden_layers,
                'activation': activation
            }

# En iyi doğruluk ve parametreler
print("En iyi doğruluk: {:.2f}".format(best_accuracy))
print("En iyi parametreler:", best_params)
 
