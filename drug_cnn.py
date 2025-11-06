import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

from tensorflow.keras import layers,models
from keras.layers import  Dense, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications import MobileNetV2
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import os

dataset="Data Combined"

image_dir=Path(dataset)

filepaths= list(image_dir.glob(r"**/*.png")) + list(image_dir.glob(r"**/*.jpg"))

labels=list(map(lambda x: os.path.split(os.path.split(x)[0])[1],filepaths))

filepaths=pd.Series(filepaths,name="filepath").astype(str)

labels=pd.Series(labels,name="label")

image_df=pd.concat([filepaths,labels],axis=1)


#Data visualization 
random_index = np.random.randint(0, len(image_df), 25)

fig, axes = plt.subplots (nrows=5, ncols=5, figsize=(11,11))


for i, ax in enumerate (axes.flat):

    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))

    ax.set_title(image_df.label[random_index[i]])

plt.tight_layout()


train_df, test_df=train_test_split(image_df,test_size=0.2,shuffle=True,random_state=42)


# data augmentation: veri attirimi

train_generator = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,

    validation_split=0.2)

test_generator = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)


train_images = train_generator.flow_from_dataframe(

        dataframe=train_df,
        x_col = "filepath", # independent -> goruntu
        y_col = "label", # dependent -> target variable -> etiket
        color_mode="rgb", class_mode="categorical",
        target_size = (224,224), # goruntulerin boyutu 
        batch_size=64,
        shuffle=True, 
        seed=42,
        subset="training" )




val_images=train_generator.flow_from_dataframe(

        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=64,
        shuffle=True,
        seed=42,
        subset="validation")



test_images=test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=(224,224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=64,
        shuffle=False)


#resize, rescale

resize_and_rescale=tf.keras.Sequential(
    [
        layers.Resizing(224, 224),
        layers.Rescaling(1./255)
    ]
) 



# transfer learning modeli (MobileNetV2), training (Fine tuning)

pretrained_model = tf.keras.applications.MobileNetV2(# onceden egitilmis model
     input_shape=(224, 224, 3), #girdilerin yani goruntulerin boyutu
     pooling = "avg",
     weights="imagenet", # hangi veri setiyle egitimi
     include_top=False, # mobilenet'in siniflandirma katmani (false katmanlarin dahil
)

pretrained_model.trainable = False # mobile neti biz train etmiyeceğiz. mobilenet zaten eğitilmiş. transsfer learning uyguluyoruz şu an. include top parametresiyle çıkartmış olduğum sınıflandırma katmanını yeniden eğiteceğim. yani transfer.
#create checkpoint callback

checkpoint_path = "checkpoint.weights.h5"

checkpoint_callback=ModelCheckpoint(checkpoint_path,
            save_weights_only= True,
            monitor="val_accuracy", 
            save_best_only=True)



# •	ModelCheckPoint: Eğitimin ilerlemesi sırasında modelin ağırlıklarını diske kaydeder.
# •	monitor="val_accuracy": Hangi metrikteki iyileşmeyi izleyeceğini belirtir. Bu durumda, doğrulama kümesindeki doğruluğu (val_accuracy) izler.
# •	save_best_only=True: Yalnızca val_accuracy değeri şimdiye kadarki en yüksek değere ulaştığında model ağırlıklarını kaydeder. Bu, eğitim sonunda en iyi performansı gösteren ağırlıklara sahip olmanızı sağlar.
# •	save_weights_only=True: Modelin tüm yapısını değil, sadece öğrenilmiş ağırlıklarını kaydeder.


early_stopping=EarlyStopping(monitor="val_loss",
                             patience=5, # 5 Epoch boyunca değişmezse durdurmaca 
                             restore_best_weights=True) #Eğitim tamalandığı zaman modelin en iyi olduğu dönemdeki ağırlıklar 


# •	EarlyStopping: Modelin gereksiz yere eğitime devam etmesini önler, böylece hem zamandan tasarruf edilir hem de aşırı öğrenme (overfitting) riski azaltılır.
# •	monitor="val_loss": Doğrulama kümesindeki kayıp (loss) değerini izler. Kayıp düşmeyi durdurursa (iyileşme durursa), durdurma işlemi başlar.
# •	patience=5: Doğrulama kaybı (val_loss) 5 çağ (epoch) boyunca iyileşmezse (düşmezse) eğitimi durdurur.
# •	restore_best_weights=True: Eğitim durdurulduğunda, en iyi val_loss değerinin elde edildiği epoch'taki ağırlıklara geri döner.


#training model - classification blok

inputs=pretrained_model.input
x=resize_and_rescale(inputs)

x=Dense(256,activation="relu")(pretrained_model.output)
x=Dropout(0.2)(x)
x=Dense(256,activation="relu")(x)
x=Dropout(0.2)(x)

outputs=Dense(10,activation="softmax")(x)

model=Model(inputs=inputs,outputs=outputs)


# •	inputs=pretrained_model.input: Birleştirilmiş modelin giriş noktası, MobileNetV2'nin beklediği giriş katmanı olarak belirlenir.
# •	x=Dense(256, activation="relu")(pretrained_model.output): Bu, MobileNetV2'nin çıktı vektörünü (önceden bahsettiğimiz pooling="avg"'dan gelen özellik vektörünü) alır ve 256 nöronlu, ReLU aktivasyonlu yeni bir tam bağlantılı (Dense) katmanına bağlar. Bu, özelliklerin yorumlanacağı ilk yeni katmandır.
# •	x=Dropout(0.2)(x): Aşırı öğrenmeyi azaltmak için, bu katmandaki nöronların %20'si rastgele devreden çıkarılır.
# •	Bu Dense ve Dropout katmanları bir kez daha tekrarlanarak modelin sınıflandırma kapasitesi artırılır.
# •	outputs=Dense(10, activation="softmax")(x): Modelin çıktı katmanıdır.
# o	10: Tahmin etmeniz gereken toplam sınıf sayısını belirtir.
# o	softmax: Çok sınıflı sınıflandırma için kullanılır. Çıktıyı, her sınıfın olasılığını temsil eden bir olasılık dağılımına dönüştürür (toplamı 1 olan bir vektör).
# •	model=Model(inputs=inputs,outputs=outputs): MobileNetV2'nin giriş katmanından (inputs) başlayıp, eklediğiniz yeni sınıflandırma katmanlarından geçerek outputs katmanında biten yeni, tam bir model oluşturulur.



model.compile(optimizer=Adam(0.0001),loss="categorical_crossentropy",metrics=["accuracy"]) 

#Adam parametre içindeki learning rate

# •	optimiser=Adam(0.0001): Modelin ağırlıklarını güncelleme algoritmasını (Adam) ve öğrenme hızını (0.0001) tanımlar. Düşük öğrenme hızı (0.0001), Transfer Öğreniminde genellikle tercih edilir.
# •	loss="categorical_crossentrophy": Modelin ne kadar hata yaptığını ölçen fonksiyondur. Etiketleriniz categorical (one-hot) formatında olduğu için bu kayıp fonksiyonu kullanılır.
# •	metrics=["accuracy"]: Eğitimi izlerken ve sonuçları değerlendirirken, doğruluk oranını (accuracy) da hesaplayıp göstermesini sağlar.


history=model.fit(train_images,
                  steps_per_epoch=len(train_images),
                  validation_data=val_images,
                  validation_steps=len(val_images),
                  epochs=10,
                  callbacks=[early_stopping,checkpoint_callback]
                  )


# •	model.fit(): Modelin eğitime başladığı ana fonksiyondur.
# •	train_images: Eğitim verisini akış halinde sağlayan ImageDataGenerator nesnesidir.
# •	steps_per_epoch=len(train_images): Bir epoch'un (tam eğitim döngüsünün) kaç adım süreceğini belirtir. Bu, ImageDataGenerator tarafından hesaplanan toplam gruplama (batch) sayısına eşittir.
# •	validation_data=val_images: Eğitimin her sonunda modelin performansını ölçmek için kullanılacak doğrulama verisini sağlar.
# •	epochs=10: Modelin tüm eğitim veri seti üzerinde 10 tam tur döneceğini belirtir (eğitimin maksimum süresi).
# •	callbacks=[early_stopping, checkpoint_callback]: Daha önce tanımlanan Erken Durdurma ve En İyi Ağırlıkları Kaydetme mekanizmalarını eğitim sürecine dahil eder.
# •	history: Eğitim sırasında kayıp ve doğruluk değerlerinin (hem eğitim hem de doğrulama için) kaydedildiği bir nesnedir, bu daha sonra grafik çizmek için kullanılır.



# Model evaluation


results=model.evaluate(test_images, verbose = 1)

print("Test loss: ",results[0])
print("Test accuracy: ",results[1])


# •	model.evaluate(test_images): Model eğitildikten sonra, tamamen ayrı tutulan test verisi üzerindeki nihai performansını hesaplar.
# •	test_images: Test verisini akış halinde sağlar (shuffle=False olarak ayarlanmıştı).
# •	Sonuçlar, test kümesi üzerindeki kayıp (loss) ve doğruluk (accuracy) olarak yazdırılır. Bu değerler, modelin genelleme yeteneğinin gerçek bir ölçümüdür.


epochs=range(1, len(history.history["accuracy"]) + 1)

hist=history.history


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, hist["accuracy"], "bo-", label="Training Accuracy")
plt.plot(epochs, hist["val_accuracy"], "r^-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, hist["loss"], "bo-", label="Training Loss")
plt.plot(epochs, hist["val_loss"], "r^-", label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




pred=model.predict(test_images)


pred=np.argmax(pred, axis=1)

labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())

pred=[labels[k] for k in pred]


random_index = np.random.randint(0, len(test_df) - 1, 15)

fig, axes=plt.subplots(nrows=5, ncols=3, figsize=(11,11))


for i, ax in enumerate(axes.flat):

    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))

    if test_df.label.iloc[random_index[i]] == pred[random_index[i]]:
        color="green"
    else:
        color="red"


    ax.set_title(f"True: {test_df.label.iloc[random_index[i]]}\n predicted: {pred[random_index[i]]}", color = color)

plt.tight_layout()

y_test=list(test_df.label)

print(classification_report(y_test, pred))