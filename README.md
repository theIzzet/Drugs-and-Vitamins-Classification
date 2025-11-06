Harika bir proje\! İlaç/Vitamin Sınıflandırma projeniz için, GitHub'a attığınızda projenizi en iyi şekilde tanıtacak, açık ve anlaşılır bir **README.md** dosyası oluşturalım.

-----

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

  * **Veri Seti Adı/Link:** [Kaggle Veri Seti Linki Buraya Eklenecek] (Örn: `https://www.kaggle.com/datasets/kandij/10-vitamin-and-drug-capsule-dataset`)

### 2\. Veri Hazırlama ve Artırma

  * **Veri Çerçevesi (DataFrame):** Görüntü dosyası yolları ve etiketleri (`label`) içeren bir Pandas DataFrame oluşturulur.
  * **Eğitim/Test Bölme:** Veri seti, eğitim ve test kümelerine ayrılır (`train_test_split`).
  * **Görüntü Veri Üreticisi (`ImageDataGenerator`):** **MobileNetV2** modelinin beklediği formatta ön işleme yapılır ve veri artırma teknikleri uygulanarak modelin genelleme yeteneği artırılır.

### 3\. MobileNetV2 Transfer Öğrenimi

  * **Temel Model:** **MobileNetV2** mimarisi, büyük bir görüntü veri seti olan **ImageNet** üzerinde eğitilmiş ağırlıklarla yüklenir.
  * **Katman Dondurma:** MobileNetV2'nin convolutional (evrişimsel) katmanları **dondurulur** (`pretrained_model.trainable = False`). Bu, önceden öğrenilmiş düşük seviyeli özellikleri korur.
  * **Sınıflandırma Başlığı:** Dondurulmuş temel modelin üzerine, bu projenin 10 sınıfına uygun yeni **Dense** (Tam Bağlantılı) katmanları eklenir. Bu katmanlar, MobileNetV2'den gelen özellikleri kullanarak ilaçları sınıflandırmayı öğrenir.
      * `Dense(256, activation="relu")` -\> `Dropout(0.2)` -\> `Dense(256, activation="relu")` -\> `Dropout(0.2)` -\> `Dense(10, activation="softmax")`

### 4\. Eğitim ve Optimizasyon

  * **Optimizer ve Kayıp Fonksiyonu:** Düşük bir öğrenme hızı (`Adam(0.0001)`) ile optimize edilir ve çok sınıflı sınıflandırma için uygun olan `categorical_crossentropy` kayıp fonksiyonu kullanılır.
  * **Callback'ler:**
      * **`ModelCheckpoint`:** En iyi doğrulama doğruluğuna (`val_accuracy`) sahip model ağırlıkları `checkpoint.weights.h5` dosyasına kaydedilir.
      * **`EarlyStopping`:** Doğrulama kaybı (`val_loss`) 5 epoch boyunca iyileşmezse (düşmezse) eğitimi durdurarak aşırı öğrenmeyi (overfitting) önler.

### 5\. Değerlendirme

Modelin performansı, ayrılan test kümesi üzerinde `model.evaluate` ve `classification_report` kullanılarak kapsamlı bir şekilde değerlendirilir.

-----

## 🛠️ Kurulum ve Çalıştırma

Projenin yerel olarak çalıştırılması en kolay yöntem, sağlanan `Dockerfile` ve `docker-compose.yml` dosyalarını kullanmaktır.

### Ön Koşullar

  * **Docker:** Sisteminize kurulu olmalıdır.
  * **Docker Compose:** Sisteminize kurulu olmalıdır (Çoğu yeni Docker kurulumu ile birlikte gelir).
  * **Model Ağırlıkları:** `checkpoint.weights.h5` dosyasının, projenin ana dizininde mevcut olması gerekmektedir. Bu dosya, `drug_cnn.py` çalıştırıldığında oluşturulur veya GitHub'dan indirilmelidir.

### Adımlar

1.  **Projeyi Klonlayın:**

    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
    cd REPO_ADINIZ
    ```

    *(`KULLANICI_ADINIZ` ve `REPO_ADINIZ` yerine kendi bilgilerinizi yazın.)*

2.  **Model Ağırlıklarını İndirin (Gerekliyse):**
    Eğer `checkpoint.weights.h5` dosyası klonlama sırasında gelmediyse (genellikle büyük dosyalar GitHub'a yüklenmez), bu dosyayı projeyi eğiterek (`drug_cnn.py` dosyasını çalıştırarak) veya projenin yayımlandığı harici bir depodan indirip ana dizine koymanız gerekmektedir.

3.  **Uygulamayı Oluşturun ve Başlatın:**
    Projenin ana dizinindeyken aşağıdaki komutu çalıştırın. Bu komut hem React arayüzünü (Vite kullanarak) oluşturacak hem de Python/FastAPI API'sini Docker konteyneri içinde başlatacaktır.

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

    *API'niz **8000** portunda çalışmasına rağmen, `docker-compose.yml` dosyasındaki yapılandırma sayesinde (port: `"80:8000"`) uygulama **80** portundan erişilebilir durumdadır.*

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

### Örnek Tahmin Yanıtı

```json
{
  "predicted_class": "Bioflu",
  "confidence": 0.985472146,
  "all_probabilities": {
    "Alaxan": 0.0001234,
    "Bactidol": 0.0000567,
    "Bioflu": 0.985472146,
    // ... diğer sınıflar
    "Neozep": 0.0000987
  },
  "success": true
}
```


