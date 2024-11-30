#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras.datasets import mnist

# MNIST verisini yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Veri boyutları ve örnek bir görüntü kontrolü
print(X_train.shape, y_train.shape)


# In[2]:


import pandas as pd
from tensorflow.keras.datasets import mnist

# MNIST verisini yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Veri boyutları ve örnek bir görüntü kontrolü
print(X_train.shape, y_train.shape)


# In[3]:


# Normalizasyon
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[4]:


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 sınıf (0-9)
])

model.summary()


# In[6]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Modeli değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


# In[7]:


import matplotlib.pyplot as plt

# Eğitim doğruluğu ve kaybı
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Eğitim kaybı
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.show()


# In[8]:


# Modeli kaydet
model.save('digit_recognizer_model.h5')

# Modeli yükle
from tensorflow.keras.models import load_model
loaded_model = load_model('digit_recognizer_model.h5')


# In[9]:


model.save('my_model.keras')


# In[13]:


import tensorflow as tf



# In[16]:


# Test verisini yükleyin (veya kendi test verinizi kullanın)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Test verisini normalize edin
x_test = x_test / 255.0

# Model ile tahminler yapın
predictions = model.predict(x_test)

# İlk birkaç tahmini yazdırın
print("Tahminler (ilk 5):", predictions[:5])

# Gerçek etiketlerle tahminlerinizi karşılaştırın
predicted_labels = np.argmax(predictions, axis=1)
print("Tahmin edilen etiketler (ilk 5):", predicted_labels[:5])
print("Gerçek etiketler (ilk 5):", y_test[:5])


# In[15]:


import numpy as np



# In[17]:


import matplotlib.pyplot as plt

# Test verisinden rastgele bir örnek seçin
sample_index = 0  # Örneğin 0. resmi seçin
image = x_test[sample_index]

# Modelin tahminini alın
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)

# Görüntüyü ve tahmin edilen sonucu gösterin
plt.imshow(image, cmap='gray')
plt.title(f"Tahmin edilen etiket: {predicted_label}")
plt.show()


# In[18]:


# Eğitim sırasında loss ve accuracy değerlerini görselleştirme
history = model.fit(x_train, y_train, epochs=10)

# Eğitim kaybı ve doğruluğu grafiği
plt.plot(history.history['accuracy'], label='Doğruluk')
plt.plot(history.history['loss'], label='Kayıp')
plt.title('Eğitim Sonuçları')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()
plt.show()


# In[19]:


from tensorflow.keras.utils import to_categorical

# Hedef etiketleri one-hot encoded formata dönüştürün
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Modeli eğitin
history = model.fit(x_train, y_train, epochs=10)


# In[20]:


# Eğitim sırasında loss ve accuracy değerlerini görselleştirme
history = model.fit(x_train, y_train, epochs=10)

# Eğitim kaybı ve doğruluğu grafiği
plt.plot(history.history['accuracy'], label='Doğruluk')
plt.plot(history.history['loss'], label='Kayıp')
plt.title('Eğitim Sonuçları')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()
plt.show()


# In[21]:


# NumPy modülünü yükleyin
import numpy as np

# TensorFlow modülünü yükleyin
import tensorflow as tf

# MNIST verisini yükleyin
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Test verisini normalize edin
x_test = x_test / 255.0

# Model ile tahminler yapın
predictions = model.predict(x_test)

# İlk birkaç tahmini yazdırın
print("Tahminler (ilk 5):", predictions[:5])

# Gerçek etiketlerle tahminlerinizi karşılaştırın
predicted_labels = np.argmax(predictions, axis=1)
print("Tahmin edilen etiketler (ilk 5):", predicted_labels[:5])
print("Gerçek etiketler (ilk 5):", y_test[:5])


# In[22]:


import matplotlib.pyplot as plt
import numpy as np

# İlk 5 tahminin görselini göster
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}")
    plt.axis('off')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

# İlk 5 tahminin görselini göster
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

# Gerçek etiketler
true_labels = y_test[:5]

plt.figure(figsize=(15, 7))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    predicted_label = predicted_labels[i]
    true_label = true_labels[i]
    
    # Renkli yazı ile tahminin doğruluğunu göster
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[24]:


import matplotlib.pyplot as plt
import numpy as np

# İlk 5 tahminin görselini göster
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

# Gerçek etiketler
true_labels = y_test[:5]

# Şık ve renkli görseller için yeni ayar
plt.figure(figsize=(15, 7))
for i in range(5):
    plt.subplot(1, 5, i+1)
    
    # Sayıyı gradyan renk paletiyle görselleştir
    plt.imshow(x_test[i], cmap='coolwarm')  # 'coolwarm' farklı renk tonları için iyi bir seçenek
    predicted_label = predicted_labels[i]
    true_label = true_labels[i]
    
    # Başlıkları daha dikkat çekici hale getirelim
    color = 'green' if predicted_label == true_label else 'red'
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", color=color, fontsize=14, weight='bold')

    # Çerçeve ekleyelim ve başlıkları etiketli hale getirelim
    plt.gca().add_patch(plt.Rectangle((0, 0), 28, 28, linewidth=3, edgecolor=color, facecolor='none'))
    
    # Sayıları gösterirken daha net yapalım
    plt.axis('off')

# Görselleştirmeyi sıkıştıralım
plt.tight_layout()
plt.show()


# In[ ]:




