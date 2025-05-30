
# 📊 Word2Vec Tabanlı Benzerlik Analizi Raporu

## 1. GİRİŞ

### Ödevin Amacı

Bu çalışmanın temel amacı, doğal dil işleme (NLP) alanında kullanılan Word2Vec modelleri ile metin benzerliği analizi yapmaktır. Farklı parametrelerle eğitilen Word2Vec modellerinin benzerlik performanslarını karşılaştırarak hangi yapıların hangi durumlarda daha etkili olduğunu tespit etmek hedeflenmiştir.

### Kullanılan Veri Seti

Çalışmada kullanılan veri seti, önceden lemmatize edilmiş 20 cümlelik kısa belgelerden oluşmaktadır. Bu veri `lemmatized_dataset.csv` adlı dosyada yer almaktadır ve her satırda bir belge bulunmaktadır. Tüm metinler anlamlı ve doğal dil kurallarına uygun cümlelerden oluşmaktadır.

---

## 2. YÖNTEM

### Benzerlik Nasıl Hesaplandı?

İki temel yöntem kullanılarak metinler arasındaki benzerlik hesaplanmıştır:

1. **TF-IDF + Kosinüs Benzerliği:**  
   - `TfidfVectorizer` kullanılarak tüm belgeler vektörleştirilmiştir.
   - Giriş cümlesi de vektörleştirilerek diğerleriyle kosinüs benzerliği hesaplanmıştır.
   - Skorlar, 0 ile 1 arasında normalize edilmiştir.

2. **Word2Vec + Ortalama Vektör + Kosinüs Benzerliği:**  
   - Her Word2Vec modeli ile belgelerdeki her kelime vektörleştirilmiş, ardından belge vektörleri ortalanmıştır.
   - Giriş cümlesi de aynı yöntemle dönüştürülmüştür.
   - Ortalama vektörler arasında kosinüs benzerliği uygulanmıştır.

### Kullanılan Modeller ve Teknikler

Toplamda 8 farklı parametre kombinasyonu kullanılarak Word2Vec modelleri eğitilmiştir. Bu kombinasyonlar şu şekildedir:

- **Model Türü (sg):**
  - `0` → CBOW
  - `1` → Skip-gram

- **Pencere Boyutu (window):** `3` ve `5`

- **Vektör Boyutu (vector_size):** `100` ve `200`

Her bir model hem `lemmatized_dataset.csv` hem de `stemmed_dataset.csv` dosyaları üzerinde eğitilmiş, yani toplam 16 model değerlendirilmiştir. Model eğitimi için Gensim kütüphanesi kullanılmıştır.

---

## 3. SONUÇ VE ÖNERİLER

### Genel Çıkarımlar

- Word2Vec modelleri, semantik anlamı yakalama konusunda TF-IDF’ye kıyasla çok daha başarılı sonuçlar vermiştir.
- Özellikle `Skip-gram` tabanlı modeller, düşük frekanslı kelimeleri daha iyi temsil ettiğinden daha doğru benzerlikler üretmiştir.
- Geniş `window` değeri kullanılan modeller daha geniş bağlamı yakalayarak anlamlı sonuçlar sunmuştur.

### Hangi Model Hangi Tür Görevler İçin Daha Uygun?

| Model               | Görev Türü                                      |
|---------------------|-------------------------------------------------|
| Skip-gram + window=5 + vector_size=200 | Anlamsal benzerlik, belge sınıflandırma |
| CBOW + window=3 + vector_size=100      | Hızlı eşleşmeler, kısa metin analizleri |
| TF-IDF                                | Basit sıralama, anahtar kelime eşleştirme |

TF-IDF yöntemi daha basit uygulamalarda tercih edilebilirken, anlam düzeyinde analiz gereken durumlarda Word2Vec modelleri önerilmektedir.

---

## ⚙️ ÇALIŞTIRMA TALİMATLARI

### Gereksinimler

Aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install pandas numpy gensim scikit-learn matplotlib tqdm
```

Python 3.7 ve üzeri önerilir.

### Dosya Gereksinimleri

- `lemmatized_dataset.csv`
- `stemmed_dataset.csv` (isteğe bağlı)
- Eğitilmiş Word2Vec modelleri (`*.model`)
- Jupyter Notebook dosyası: `similarity_analysis.ipynb`  
  veya Python script’i: `main.py`

### Kullanım Adımları

1. **Notebook Ortamını Başlatın**

```bash
jupyter notebook
```

2. **Notebook'u Açın ve Çalıştırın**  
   `similarity_analysis.ipynb` dosyasını açın. Giriş cümlesini tanımlayın:

```python
input_text = "Afetlere hazırlıklı olmak toplumsal bir sorumluluktur."
```

3. **Tüm hücreleri sırayla çalıştırın.**  
   Sonuçları hem grafiklerle hem de tablolarla inceleyebilirsiniz.

4. (Opsiyonel) Komut satırından çalıştırmak isterseniz:

```bash
python main.py
```

Giriş cümlesini script üzerinden değiştirerek farklı analizler de yapabilirsiniz.

---

## 📌 Ek Notlar

- Her modelin çıktı skoru ve en benzer belge bilgisi ayrı ayrı kaydedilmiştir.
- Çalışma kapsamında sadece lemmatized veri kullanılmıştır, ancak kodlar stemmed veriyle de çalışacak şekilde hazırlanmıştır.
- Daha büyük veri kümeleriyle test edilerek model genellemesi yapılabilir.

---

Hazırlayan: **Semanur**  
Tarih: **Mayıs 2025**  

