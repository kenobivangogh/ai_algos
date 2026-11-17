import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

def load_data(file_path):
    """Veri setini yükler."""
    return pd.read_csv(file_path)

def fill_missing_values(df):
    """Eksik değerleri doldurur."""
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Sayısal eksik değerleri ortalama ile doldurma
    df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.mean()))

    # Kategorik eksik değerleri mod ile doldurma
    df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
    return df

def normalize_data(df):
    """Sayısal değişkenleri 0-1 aralığında normalize eder."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def encode_categorical(df):
    """Kategorik değişkenleri sayısal değerlere dönüştürür."""
    categorical_cols = df.select_dtypes(include=["object"]).columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])
    return df

def split_data(df, target_col, test_size=0.2):
    """Veriyi train-test setlerine böler."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def perform_grid_search(X_train, y_train):
    """MLPClassifier için GridSearchCV işlemini gerçekleştirir."""

# "ConvergenceWarning: Stochastic Optimizer: Maximum iterations reached and the optimization hasn't converged yet." uyarısını veriyordu sürekli
# bu satır ile uyarıyı görmezden gelerek kod çalışmaya devam ediyor, aksi halde terminalde çok fazla bir uyarı yığını oluşuyordu.
    warnings.filterwarnings("ignore", category=ConvergenceWarning) 

    param_grid = {
        "hidden_layer_sizes": [
            (10,), (20,), (30,),                 # 1 katman
            (10, 10), (20, 20), (30, 30),       # 2 katman
            (10, 10, 10), (20, 20, 20), (30, 30, 30),  # 3 katman
            (10, 10, 10, 10), (20, 20, 20, 20), (30, 30, 30, 30)  # 4 katman
        ],
        "activation": ["logistic", "relu", "tanh"],
        "max_iter": [200, 500]
    }
    grid_search = GridSearchCV(
        estimator=MLPClassifier(random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(grid_search, X_test, y_test):
    """En iyi modeli test veri seti üzerinde değerlendirir."""
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, grid_search.best_params_

# Adım 1-10
def steps_1_to_10(file_path, target_col):
    # 1.-5. adımlar
    df = load_data(file_path)
    df = fill_missing_values(df)
    df = normalize_data(df)
    df = encode_categorical(df)

    # 6. adım
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    # 7.-8. adımlar
    grid_search = perform_grid_search(X_train, y_train)

    # 9.-10. adımlar
    accuracy_filled, best_params_filled = evaluate_model(grid_search, X_test, y_test)

    print(f"Test Accuracy: {accuracy_filled}")
    print(f"Best Parameters: {best_params_filled}")

    return accuracy_filled, best_params_filled

# Adım 4-10
def steps_4_to_10(file_path, target_col):
    # Veri setini yükler ve eksik değerleri çıkarır
    df = load_data(file_path)
    df = df.dropna()
    df = normalize_data(df)
    df = encode_categorical(df)

    # 6. adım
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    # 7.-8. adımlar
    grid_search = perform_grid_search(X_train, y_train)

    # 9.-10. adımlar
    accuracy_dropped, best_params_dropped = evaluate_model(grid_search, X_test, y_test)

    print(f"Test Accuracy: {accuracy_dropped}")
    print(f"Best Parameters: {best_params_dropped}")

    return accuracy_dropped, best_params_dropped

# Karşılaştırma
def main():
    file_path = "hepatitis.csv"  
    target_col = "Class"  

    print("Eksik veriler doldurularak işlemler başlatılıyor...")
    accuracy_filled, best_params_filled = steps_1_to_10(file_path, target_col)

    print("\nEksik veriler çıkarılarak işlemler başlatılıyor...")
    accuracy_dropped, best_params_dropped = steps_4_to_10(file_path, target_col)

    # Yorumlama
    print("\n# Karşılaştırma ve Yorumlar")
    if accuracy_filled > accuracy_dropped:
        print("""
        Eksik veriler doldurulduğunda model daha yüksek doğruluk sağladı.
        - Doğruluk oranı: {accuracy_filled:.3f} (Eksik veriler dolduruldu)
        - Doğruluk oranı: {accuracy_dropped:.3f} (Eksik veriler çıkarıldı)
        
        Yorum:
        Eksik veriler doldurulduğunda, model daha fazla bilgiye erişmiş ve bu bilgi modeli daha iyi eğitmiştir.
              
        Avantaj: Veri setini büyütür, daha fazla veri ile model eğitimi yapılır.
        Dezavantaj: Eğer doldurma işlemi yanlış veya eksikse, modelin performansını olumsuz etkileyebilir.

        Eksik verilerin doldurulması, orijinal veriye dayalı yapay bilgi ekler. Bu, modelin bazı durumlarda yanlış ilişkilendirmeler yapmasına neden olabilir.
        """.format(accuracy_filled, accuracy_dropped))            
        
    else:
        print("""
        Eksik veriler çıkarıldığında model daha yüksek doğruluk sağladı.
        - Doğruluk oranı: {:.3f} (Eksik veriler çıkarıldı)
        - Doğruluk oranı: {:.3f} (Eksik veriler dolduruldu)

        Yorum:
        Eksik veriler çıkarıldığında, model yalnızca mevcut ve güvenilir verilere dayanmıştır.

        Avantaj: Orijinal verilere dayanır, yanlış bilgi eklenmez.
        Dezavantaj: Veri seti küçülür, öğrenme için yeterli bilgi olmayabilir.

        Eksik verilerin çıkarılması, veri setini küçültür ve modelin daha basit bir yapıyla genel özellikleri öğrenmesine olanak tanır. Ancak, eksik veriler çıkarıldığında veri seti küçüldüğü için model bazı önemli bilgileri kaybedebilir.
        """.format(accuracy_dropped, accuracy_filled))

if __name__ == "__main__":
    main()