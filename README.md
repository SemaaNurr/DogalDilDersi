
# 📊 Doğal Dil İşleme ile Metin Analizi

📌 **Proje Sahibi:** Semanur  


---

## 📌 Proje Tanımı

Bu proje, **doğal dil işleme (NLP)** teknikleri kullanılarak metin tabanlı veri analizi yapmayı amaçlamaktadır. Çalışmada **yemek tarifleri** içeren bir veri seti üzerinden **Zipf Yasası**, **TF-IDF** ve **Word2Vec** yöntemleriyle metin analizi gerçekleştirilmiştir.

---

## 📊 Veri Seti Detayları

- **Kaynak:** [Kaggle – Food Recipes Dataset](https://www.kaggle.com/)
- **Kullanılan Kaynak** https://www.kaggle.com/datasets/semanurkeskin/food-dataset
- **Boyut:** 37.737 satır, 36.1 MB (CSV formatında)  
- **Kullanım Amacı:**
  - 📌 Zipf Yasası ile kelime frekanslarını incelemek  
  - 📌 TF-IDF vektörleştirme ile kelime önem derecelerini belirlemek  
  - 📌 Word2Vec modeli ile kelime ilişkilerini öğrenmek  

---

## ⚙️ Adım Adım Model Oluşturma

### 🔹 1️⃣ Veri Seti Yükleme

```python
import pandas as pd

# CSV dosyasını yükleyelim
file_path = "food-dataset-en.csv"
df = pd.read_csv(file_path)

# Veri içeriğine göz atalım
print(df.head())
```

---

### 🔹 2️⃣ Ön İşleme (Pre-processing)

Veri temizleme ve kelime işlemleri için:

- Gereksiz sütunlar kaldırıldı
- Stop words temizlendi
- Tokenization ve lowercase işlemleri uygulandı
- Lemmatization & Stemming yapıldı  

```python
import nltk
from nltk.corpus import stopwords
import re

# Stop words listesini yükleyelim
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Temizlenmiş metin ekleyelim
df["cleaned_text"] = df["text"].apply(clean_text)

print(df.head())
```

---

### 🔹 3️⃣ Zipf Yasası Analizi

Kelime frekanslarını hesaplayıp **log-log grafiği** ile Zipf Yasası incelendi.

```python
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Kelime frekanslarını hesaplayalım
words = " ".join(df["cleaned_text"]).split()
word_counts = Counter(words)

sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
words, freqs = zip(*sorted_word_counts)

# Log-log grafiği
plt.figure(figsize=(8, 6))
log_freqs = np.log(freqs)
log_ranks = np.log(range(1, len(freqs) + 1))

plt.plot(log_ranks, log_freqs, marker="o", linestyle="None")
plt.title("Zipf Yasası - Log-Log Grafiği")
plt.xlabel("Log(Sıra)")
plt.ylabel("Log(Frekans)")
plt.grid(True)
plt.show()
```

---

### 🔹 4️⃣ TF-IDF Vektörleştirme

Metinlerdeki kelime önem derecelerini **TF-IDF** yöntemiyle belirleme.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"])

# DataFrame oluşturma
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf.to_csv("tfidf_lemmatized.csv", index=False)
```

---

### 🔹 5️⃣ Word2Vec Model Eğitimi

Kelime ilişkilerini öğrenmek için **Word2Vec** modeli eğitildi.

```python
from gensim.models import Word2Vec

# Tokenize edilmiş kelimeler
sentences = [row.split() for row in df["cleaned_text"]]

# Word2Vec modeli eğitme
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
model.save("word2vec_lemmatized.model")

print(model.wv.most_similar("tarif"))
```

---

## 📦 Gerekli Kütüphaneler ve Kurulum

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerinin kurulması gerekmektedir:

```bash
pip install pandas numpy nltk matplotlib scikit-learn gensim
```

- **pandas** → Veri yükleme ve temizleme  
- **numpy** → Sayısal işlemler  
- **nltk** → Kelime temizleme ve metin işlemleri  
- **matplotlib** → Görselleştirme  
- **scikit-learn** → TF-IDF uygulamaları  
- **gensim** → Word2Vec modeli eğitimi  

---

## 📌 Notlar

- Veri seti İngilizce yemek tariflerinden oluşmaktadır.
- Proje dosyaları içerisinde ilgili Python kodları ve çıktı CSV dosyaları yer almaktadır.
- Word2Vec çıktısı olarak belirli kelimelerin benzer kelimeleri örnek olarak verilmiştir.

---

## 📬 İletişim

Herhangi bir soru ve geri bildirim için proje sahibiyle iletişime geçebilirsiniz.
