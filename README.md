# 📊 Akbank Makine Öğrenmesi Bootcamp Projesi – **Airbnb Fiyat Tahmini**

**Kaggle Notebook:** <https://www.kaggle.com/code/ayemdamlakeskin/gaih-ml-25?scriptVersionId=241819790>  
**Kaggle Veri Seti:** <https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata>  

## İçindekiler
1. [Proje Tanıtımı](#proje-tanıtımı)  
2. [Veri Seti](#veri-seti)  
3. [Kurulum ve Gereksinimler](#kurulum-ve-gereksinimler)  
4. [Keşifsel Veri Analizi (EDA)](#keşifsel-veri-analizi-eda)  
5. [Veri Ön İşleme](#veri-ön-işleme)  
6. [Modelleme ve Hiperparametre Optimizasyonu](#modelleme-ve-hiperparametre-optimizasyonu)  
7. [Model Değerlendirme](#model-değerlendirme)  
8. [Sonuçlar](#sonuçlar)  
9. [Gelecekteki İyileştirmeler](#gelecekteki-iyileştirmeler)  

---

## Proje Tanıtımı
Bu projede, **Airbnb ilanlarının gecelik fiyatlarını** tahmin etmek amacıyla gözetimli öğrenme algoritmaları uygulandı. Model, **regresyon** problemini çözerek:

* Ev sahiplerinin ilanlarını doğru fiyatlamasına,
* Platformun gelir optimizasyonuna,
* Misafirlerin bütçelerine uygun seçenek bulmasına katma değer sağlamayı hedefler.

|                     |                                         |
|---------------------|-----------------------------------------|
| **Hedef Değişken**  | `PRICE` (USD, float)                    |
| **Özellik Sayısı**  | 25 (temizleme sonrası)                  |
| **Gözlem Sayısı**   | 96 199                                  |
| **Problem Türü**    | Regresyon                               |

---

## Veri Seti
| Özellik                 | Açıklama                                                     |
|-------------------------|--------------------------------------------------------------|
| `LAT`, `LONG`           | Enlem & boylam                                              |
| `ROOM TYPE`             | Oda/konut tipi (Entire home/apt, Private room, …)           |
| `MINIMUM NIGHTS`        | Minimum konaklama gecesi                                    |
| `AVAILABILITY 365`      | Yıl içindeki müsait gün sayısı                              |
| `NUMBER OF REVIEWS`     | Toplam yorum adedi                                          |
| `REVIEW RATE NUMBER`    | Ortalama puan (1–5)                                         |
| …                       | … (toplam 25 değişken)                                      |

*Kaynak:* Kaggle – *Airbnb Open Data*  
*Boyut:* 150 MB   /    *Gözlem:* > 100 000

---

## Kurulum ve Gereksinimler
```bash
# Sanal ortam (önerilir)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Gereksinimler
pip install -r requirements.txt
````

**Ana Paketler:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `folium`

---

## Keşifsel Veri Analizi (EDA)

EDA aşamasında:

* Eksik & aykırı değer analizi (missingno, boxplot)
* Oda tipi–fiyat ilişkisi, mahalle bazlı dağılımlar
* Fiyat–servis ücreti korelasyonu (ρ ≈ 1.0)
* Minimum konaklama, müsaitlik süresi, yorum yoğunluğu
* Korelasyon matrisi & PCA ile boyut indirgeme
  → PC1: oda tipi + minimum gece, PC2: mahalle + müsaitlik

Detaylı kod ve görseller için Kaggle defterine göz atabilirsiniz.

---

## Veri Ön İşleme

| Adım                 | Yöntem / Not                                        |
| -------------------- | --------------------------------------------------- |
| Sütun İsim Düzenleme | Hepsi BÜYÜK HARF                                    |
| Gereksiz Sütunlar    | `LICENSE`, `HOUSE_RULES`, … atıldı                  |
| Eksik Değer          | Satır bazlı silme (kalan % 94)                      |
| Aykırı Değer         | `MINIMUM_NIGHTS`, `AVAILABILITY_365` → clip(1, 365) |
| Metin Temizliği      | `$` kaldırma, string lower-case                     |
| Kategorik Kodlama    | `LabelEncoder` + `pd.get_dummies`                   |
| Veri Bölme           | Eğitim %80 – Test %20 (shuffle)                     |

---

## Modelleme ve Hiperparametre Optimizasyonu

| Algoritma                   | Neden Seçildi?                      |
| --------------------------- | ----------------------------------- |
| **Random Forest Regressor** | Non-lineer ilişkiler & feature önemli |
| XGBoost Regressor           | Boosting ile yüksek doğruluk        |
| HistGradientBoostingReg.    | Büyük veri & hız                    |

İlk denemelerde **Random Forest** en yüksek R² verdiği için hiperparametre optimizasyonu (GridSearchCV) bu modelde yapıldı.

```python
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "max_features": [0.5, "sqrt", "log2"],
    "bootstrap": [True, False]
}
gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    params,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
)
gs.fit(X_train, y_train)
```

**En İyi Parametreler:**
`{'n_estimators': 300, 'max_depth': 20, 'max_features': 0.5, 'bootstrap': True}`

---

## Model Değerlendirme

**Test Sonuçları – Random Forest**

| Metric           | Değer         |
| ---------------- | ------------- |
| **R² (Test)**    | **0.819**     |
| RMSE             | **83.76 USD** |
| MSE              | 7 015.23 USD² |
| Negatif MSE (CV) | –8 406.15     |

> En iyi hiperparametre kümesi ile (n\_estimators = 300, max\_depth = 20, max\_features = 0.5, …) model, test verisinde **%82 doğruluk** ve **±84 USD** ortalama hata payına ulaştı.

---

## Sonuçlar

* Model, New York Airbnb ilanlarında gecelik fiyatı **≈ ±84 USD hata payı** ile tahmin edebildi.
* Oda tipi ve servis ücreti fiyatın en güçlü belirleyicileri olarak öne çıktı.
* Mahalle & konum bilgileri ham haliyle (lat/long) ölçülü katkı sağladı; mekânsal zenginleştirme ihtiyacı net görüldü.

---

## Gelecekteki İyileştirmeler

* **Zaman Serisi Özellikleri:** İlan tarihi, sezon, etkinlik takvimi (tatil, konferans) gibi zamana bağlı değişkenler eklemek.
* **Fiyat Aykırı Değer İşleme:** Robust scaler, winsorization veya quantile loss regression ile uç noktaların etkisini azaltmak.
* **Model Karma:** Gradient Boosting + Neural Network ensemble ile doğruluk artışı.
* **Explainable AI:** SHAP / LIME ile ev sahiplerine fiyat tavsiyesini açıklamak.
* **API & Dashboard:** FastAPI + Streamlit panel ile gerçek zamanlı fiyat öneri servisi.
* **Otomatik Veri Boru Hattı:** Airflow destekli günlük veri çekme, eğitim ve model izleme (drift tespiti).
