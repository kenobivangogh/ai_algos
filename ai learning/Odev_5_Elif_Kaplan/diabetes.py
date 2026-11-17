# Elif Kaplan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv("diabetes.csv")

X = data.drop(columns=["Outcome"]) # veri setinden "Outcome" sütununu çıkarıp kalan tüm sütunları X'e at
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = ["linear", "rbf", "sigmoid", "poly"]
C_values = [0.1, 1, 10] 

best_accuracy = 0
best_kernel = None
best_C = None

# Tüm kernel ve C kombinasyonları ile modeller oluşturup test etme
print("# Kernel   C    Accuracy")
print("# -----   ---   --------")
for kernel in kernels:
    for C in C_values:
        if kernel == "poly":
            # poly kernelinde işlem yükü ve çalışma süresi uzun sürdü
            # bu yüzden poly kernelinde kullanılan polinom derecesini düşürdüm, hızlı çalıştı
            model = SVC(kernel=kernel, C=C, degree=2)
        else:
            model = SVC(kernel=kernel, C=C, gamma="auto" if kernel != "linear" else "scale")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{kernel:<7} {C:<5} {accuracy:.2f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel
            best_C = C

print("\nEn yüksek doğruluğu veren model:")
print(f"Kernel: {best_kernel}, C: {best_C}, Accuracy: {best_accuracy:.2f}")
print("\nBüyük C değerleri zeta katsayısının etkisini azaltır")
