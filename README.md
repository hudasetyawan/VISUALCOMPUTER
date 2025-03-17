# Model-Warna-pada-citra
Pengujian Model Warna Pada citra
```python
# Image Processing Techniques on Google Colab 

from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Upload image file
uploaded = files.upload()

# Membaca gambar yang diunggah
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fungsi untuk menampilkan gambar asli dan hasil proses

def show_images(title1, image1, title2, image2):
    plt.figure(figsize=(12,6))

    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.show()

# Menampilkan Gambar Asli
show_images('Gambar Asli (Grayscale)', image_gray, 'Gambar Asli (RGB)', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 1. Citra Negative
negative_image = 255 - image_gray
show_images('Gambar Asli', image_gray, 'Citra Negative', negative_image)

# 2. Transformasi Log
c = 255 / np.log(1 + np.max(image_gray))
log_image = c * np.log(1 + image_gray.astype(np.float32))
log_image = np.uint8(log_image)
show_images('Gambar Asli', image_gray, 'Transformasi Log', log_image)

# 3. Transformasi Power Law (Gamma Correction)
gamma = 0.5  # Ubah nilai gamma untuk hasil berbeda
power_law_image = np.array(255 * (image_gray / 255) ** gamma, dtype='uint8')
show_images('Gambar Asli', image_gray, 'Transformasi Power Law', power_law_image)

# 4. Histogram Equalization
equalized_image = cv2.equalizeHist(image_gray)
show_images('Gambar Asli', image_gray, 'Histogram Equalization', equalized_image)

# 5. Histogram Normalization
normalized_image = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
show_images('Gambar Asli', image_gray, 'Histogram Normalization', normalized_image)

# 6. Konversi RGB ke HSI
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsi = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
show_images('Gambar Asli (RGB)', image_rgb, 'Konversi RGB ke HSI (Hue Component)', image_hsi[:,:,0])

# Menentukan thresholding (contoh sederhana)
threshold_value = 120
_, threshold_image = cv2.threshold(image_hsi[:,:,2], threshold_value, 255, cv2.THRESH_BINARY)
show_images('Gambar Asli (Intensity Component)', image_hsi[:,:,2], 'Thresholding pada Komponen Intensity', threshold_image)
```
GAMBAR ORIGINAL PENGUJIAN : Kota Berlin,Jerman
![image](https://github.com/user-attachments/assets/1456b305-2da2-4c95-8bc9-e322ed9d44fa)

1. Citra Negative
Input: Gambar dalam skala abu-abu (grayscale).
Output: Gambar negatif dari input dengan menginversikan nilai pixel (255 - pixel value).
Penggunaan: Cocok untuk meningkatkan visibilitas pada gambar gelap atau mendeteksi fitur tertentu.
![image](https://github.com/user-attachments/assets/79c4e995-e870-44f9-aaf0-609ae755b235)

2. Transformasi Log
Input: Gambar dalam skala abu-abu (grayscale).
Output: Gambar dengan intensitas pixel terkompresi secara logaritmik, menyoroti area gelap tanpa menghilangkan detail di area terang.
Penggunaan: Berguna untuk meningkatkan kontras dari gambar yang sangat gelap atau terlalu terang.
![image](https://github.com/user-attachments/assets/9f280089-387e-4575-a08e-d5dbe1c336bf)

 3. Transformasi Power Law (Gamma Correction)
Input: Gambar dalam skala abu-abu (grayscale) dan nilai gamma (misalnya 0.5 atau 2.0).
Output: Gambar yang telah diubah tingkat kecerahannya dengan rumus 
𝑠
=
𝑐
×
𝑟
𝛾
s=c×r 
γ
 .
Penggunaan: Memperbaiki gambar yang terlalu terang atau terlalu gelap. Nilai gamma < 1 mencerahkan gambar, sedangkan nilai gamma > 1 menggelapkannya.
![image](https://github.com/user-attachments/assets/5e3bb5b0-68b2-48fc-8151-604ab8b7bd0b)

4. Histogram Equalization
Input: Gambar dalam skala abu-abu (grayscale).
Output: Gambar dengan distribusi intensitas pixel yang lebih merata, meningkatkan kontras keseluruhan.
Penggunaan: Cocok untuk gambar dengan kontras rendah yang perlu ditingkatkan.
![image](https://github.com/user-attachments/assets/ed23ba6f-fcc8-4002-a963-94e1669fd33d)

5. Histogram Normalization
Input: Gambar dalam skala abu-abu (grayscale).
Output: Gambar dengan rentang intensitas pixel yang diubah agar memenuhi seluruh rentang 0 - 255.
Penggunaan: Berguna untuk menormalkan pencahayaan yang tidak merata.
![image](https://github.com/user-attachments/assets/45bb0a32-0620-4321-90c9-5a4621730016)

 6. Konversi RGB ke HSI & Thresholding
Input: Gambar asli dalam format RGB.
Output: Gambar dalam format HSI (Hue, Saturation, Intensity), kemudian dikonversi menjadi gambar biner menggunakan metode thresholding.
Penentuan Thresholding:
Thresholding dilakukan pada komponen Intensity (I) karena lebih baik merepresentasikan tingkat kecerahan gambar.
Misalkan kita memiliki gambar grayscale dengan nilai pixel dari 0 (hitam) hingga 255 (putih). Jika kita memilih nilai threshold = 120, maka setiap pixel dengan nilai lebih besar dari 120 akan dianggap sebagai putih (255), sedangkan yang lainnya akan dianggap hitam (0).
Penggunaan: Deteksi objek atau segmentasi gambar dengan pencahayaan seragam.
![image](https://github.com/user-attachments/assets/071073d1-e5c1-4f30-acd7-557f577f45c7)



