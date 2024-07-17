import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Muat dataset
url = 'https://drive.google.com/uc?id=184rRE0pEKmD_ztSAFFoaTYyXeUNiEdRd'
df = pd.read_csv(url)

# Pisahkan fitur dan target
X = df.drop(columns=['id', 'cardio'])
y = df['cardio']

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Latih model
knn.fit(X_train, y_train)

# Simpan model dan scaler
with open('model/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
 