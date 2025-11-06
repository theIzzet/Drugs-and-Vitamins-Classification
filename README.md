# 💊 İlaç/Vitamin Görüntü Sınıflandırma API'si (MobileNetV2 Transfer Öğrenimi)

Bu proje, bir Görüntü İşleme (Computer Vision) modeli kullanarak 10 farklı ilaç ve vitamin türünü sınıflandıran bir **Convolutional Neural Network (CNN)** uygulamasını içerir. Model eğitimi için **Transfer Öğrenimi** metodolojisi, sunumu için ise **FastAPI** ve **React** kullanılarak tam yığın (Full-Stack) bir uygulama geliştirilmiştir. Uygulama, **Docker** ve **Docker Compose** kullanılarak kolayca dağıtılabilir hale getirilmiştir.

## 🚀 Kullanılan Teknolojiler

### Makine Öğrenimi / Yapay Zeka

  * **TensorFlow / Keras:** Derin öğrenme modeli (CNN) inşası ve eğitimi için temel çerçeve.
  * **MobileNetV2:** Modelin temelini oluşturan, görüntü tanıma görevleri için önceden eğitilmiş hafif bir CNN mimarisi. (**Transfer Öğrenimi** için kullanılmıştır.)
  * **NumPy, Pandas, Matplotlib:** Veri işleme, görselleştirme ve metrik hesaplama için kullanılan temel bilimsel kütüphaneler.

### Arka Uç (Backend) / API

  * **Python:** Projenin temel geliştirme dili.
  * **FastAPI:** Hızlı, modern, Python tabanlı ve otomatik dokümantasyonlu bir API oluşturmak için kullanılmıştır. Görüntü yükleme ve tahmin uç noktalarını yönetir.
  * **Uvicorn:** FastAPI uygulamasını asenkron olarak çalıştırmak için kullanılan ASGI sunucusu.
  * **Pillow (PIL):** Görüntüleri işlemek ve modele hazır hale getirmek için kullanılmıştır.

### Ön Uç (Frontend) / Kullanıcı Arayüzü

  * **React:** Kullanıcı arayüzünü oluşturmak için kullanılan popüler JavaScript kütüphanesi.
  * **Vite:** Hızlı ve modern ön uç geliştirme için kullanılan bir derleme aracı.

### Dağıtım (Deployment)

  * **Docker:** Uygulamanın ve tüm bağımlılıklarının (hem Python API'si hem de React arayüzü) konteynerize edilmesini sağlar.
  * **Docker Compose:** Uygulamanın tek bir komutla ayağa kaldırılmasını kolaylaştırır.

-----

## 💻 Model Eğitim Metodolojisi (`drug_cnn.py`)

Projede, verimli ve başarılı bir sınıflandırma modeli oluşturmak için **Transfer Öğrenimi** yaklaşımı benimsenmiştir.

### 1\. Veri Seti

Bu proje için kullanılan veri seti Kaggle platformundan alınmıştır:

  * **Kullanılan Veri Seti :** https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images

### 2\. Veri Hazırlama ve Artırma

  * **Veri Çerçevesi (DataFrame):** Görüntü dosyası yolları ve etiketleri (`label`) içeren bir Pandas DataFrame oluşturulur.
  * **Eğitim/Test Bölme:** Veri seti, eğitim ve test kümelerine ayrılır (`train_test_split`).
  * **Görüntü Veri Üreticisi (`ImageDataGenerator`):** **MobileNetV2** modelinin beklediği formatta ön işleme yapılır ve veri artırma teknikleri uygulanarak modelin genelleme yeteneği artırılır.

### 3\. MobileNetV2 Transfer Öğrenimi

  * **Temel Model:** **MobileNetV2** mimarisi, büyük bir görüntü veri seti olan **ImageNet** üzerinde eğitilmiş ağırlıklarla yüklenir.
  * **Katman Dondurma:** MobileNetV2'nin convolutional (evrişimsel) katmanları **dondurulur** (`pretrained_model.trainable = False`). Bu, önceden öğrenilmiş düşük seviyeli özellikleri korur.
  * **Sınıflandırma Başlığı:** Dondurulmuş temel modelin üzerine, bu projenin 10 sınıfına uygun yeni **Dense** (Tam Bağlantılı) katmanları eklenir.

### 4\. Eğitim ve Optimizasyon

  * **Optimizer ve Kayıp Fonksiyonu:** Düşük bir öğrenme hızı (`Adam(0.0001)`) ile optimize edilir ve çok sınıflı sınıflandırma için uygun olan `categorical_crossentropy` kayıp fonksiyonu kullanılır.
  * **Callback'ler:**
      * **`ModelCheckpoint`:** En iyi doğrulama doğruluğuna (`val_accuracy`) sahip model ağırlıkları `checkpoint.weights.h5` dosyasına kaydedilir.
      * **`EarlyStopping`:** Doğrulama kaybı (`val_loss`) 5 epoch boyunca iyileşmezse (düşmezse) eğitimi durdurarak aşırı öğrenmeyi (overfitting) önler.

-----

## 📈 Model Performansı ve Sonuçlar

Eğitim sürecinde elde edilen performans metrikleri ve grafikler aşağıdadır. Model, 10 epoch boyunca eğitilmiş ve test kümesinde yüksek bir doğruluk oranı elde etmiştir.

### Eğitim ve Doğrulama Grafikleri

Aşağıdaki grafikler, modelin eğitim ve doğrulama veri setlerindeki **Doğruluk (Accuracy)** ve **Kayıp (Loss)** değerlerinin 10 epoch boyunca nasıl değiştiğini göstermektedir.

  * **Doğruluk Grafiği:** Eğitim doğruluğu sürekli yükselirken, doğrulama doğruluğu da 5. epoktan sonra yavaşlayarak yaklaşık **%84** seviyelerinde dengelenmiştir. Bu, modelin genelleme yeteneğinin iyi olduğunu gösterir.
  * **Kayıp Grafiği:** Hem eğitim hem de doğrulama kayıpları istikrarlı bir şekilde düşerek modelin öğrenme sürecinin başarılı olduğunu göstermektedir.

### Nihai Test Sonuçları

Eğitimden sonra, modelin daha önce görmediği test veri kümesi üzerindeki sonuçları:

| Metrik | Değer |
| :--- | :--- |
| **Test Kaybı (Loss)** | **0.469** |
| **Test Doğruluğu (Accuracy)** | **%83.60** |

### Sınıflandırma Raporu (Test Kümesi)

Modelin her bir ilaç sınıfı için gösterdiği performans (Hassasiyet, Geri Çağırma, F1-Skoru):

```
              precision    recall  f1-score   support

     Alaxan       0.83      0.88      0.85       208
   Bactidol       0.85      0.77      0.81       202
     Bioflu       0.91      0.81      0.86       192
   Biogesic       0.84      0.71      0.77       201
    DayZinc       0.91      0.84      0.87       209
   Decolgen       0.88      0.87      0.87       186
   Fish Oil       0.90      0.90      0.90       211
   Kremil S       0.69      0.85      0.76       204
    Medicol       0.89      0.91      0.90       212
     Neozep       0.71      0.83      0.77       175

   accuracy                           0.84      2000
  macro avg       0.84      0.84      0.84      2000
weighted avg      0.84      0.84      0.84      2000
```

  * **Genel Performans:** Model, 10 farklı ilaç sınıfını ayırt etmede ortalama **%84** doğruluk (accuracy) ile iyi bir performans sergilemiştir.
  * **Öne Çıkanlar:** `Bioflu`, `DayZinc`, `Fish Oil` ve `Medicol` gibi sınıflarda $\ge 0.86$ F1-Skoru ile en yüksek performansı göstermiştir.
  * **Geliştirilebilecek Alanlar:** `Kremil S` ve `Neozep` sınıfları, diğerlerine göre daha düşük **Precision** değerlerine sahip olup, potansiyel olarak geliştirme veya daha fazla veri toplama gerektirebilir.

-----

## 🛠️ Kurulum ve Çalıştırma

Projenin yerel olarak çalıştırılması en kolay yöntem, sağlanan `Dockerfile` ve `docker-compose.yml` dosyalarını kullanmaktır.

### Ön Koşullar

  * **Docker:** Sisteminize kurulu olmalıdır.
  * **Docker Compose:** Sisteminize kurulu olmalıdır (Çoğu yeni Docker kurulumu ile birlikte gelir).
  * **Model Ağırlıkları:** `checkpoint.weights.h5` dosyasının, projenin ana dizininde mevcut olması gerekmektedir.

### Adımlar

1.  **Projeyi Klonlayın:**

    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
    cd REPO_ADINIZ
    ```

    *(`KULLANICI_ADINIZ` ve `REPO_ADINIZ` yerine kendi bilgilerinizi yazın.)*

2.  **Model Ağırlıklarını İndirin (Gerekliyse):**
    `checkpoint.weights.h5` dosyasını (eğitilmiş model ağırlıkları), projenin ana dizinine yerleştirmeniz gerekmektedir.

3.  **Uygulamayı Oluşturun ve Başlatın:**
    Projenin ana dizinindeyken aşağıdaki komutu çalıştırın.

    ```bash
    docker-compose up --build -d
    ```

      * `--build`: İlk çalıştırmada imajları oluşturur.
      * `-d`: Konteyneri arka planda (detached) çalıştırır.

4.  **Uygulamaya Erişin:**
    Uygulama başarıyla başlatıldıktan sonra, web tarayıcınızda aşağıdaki adrese gidin:

    ```
    http://localhost:80
    ```

### Kapatma

Uygulamayı durdurmak ve konteyneri kaldırmak için:

```bash
docker-compose down
```

-----

## 📝 API Uç Noktaları (`main.py`)

FastAPI uygulamanız, tahmin işlemini gerçekleştirmek ve sınıf isimlerini sağlamak için aşağıdaki uç noktalarını sunar:

| Metot | Uç Noktası | Açıklama |
| :---: | :---: | :--- |
| `GET` | `/` | React ön yüzünü (static/index.html) sunar. |
| `GET` | `/health` | Uygulama sağlığı ve modelin yüklenip yüklenmediği bilgisini verir. |
| `GET` | `/classes` | Modelin sınıflandırdığı 10 ilacın/vitaminin isimlerini listeler. |
| `POST` | `/api/predict` | Yüklenen bir görüntü dosyasını alır ve sınıflandırma tahminini döndürür. |




