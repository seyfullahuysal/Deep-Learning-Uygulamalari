#KULLANIM:  python opencv_derin_ogrenme.py --resim ../araba.jpg
# Gerekli paketler
import numpy as np
import argparse
import time
import cv2

#diskteki dosyalar
MODEL ="bvlc_googlenet.caffemodel"
PROTOTXT="bvlc_googlenet.prototxt"
SINIFLAR="synset_words.txt"

#gerekli argümanların alımı
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--resim", required=True,help="--resim ile resim yolu gosterin")
argumanlar = vars(ap.parse_args())

# Resim yükleme
resim = cv2.imread(argumanlar["resim"])

# modelde tanımlı olan sınıfları çekme
satir = open(SINIFLAR).read().strip().split("\n")
siniflar = [r[r.find(" ") + 1:].split(",")[0] for r in satir]


# Resim 224x224 olarak yeniden boyutlandırıldı, Mean subtraction yapılarak blob çıkarma yapıldı

blob = cv2.dnn.blobFromImage(resim, 1, (224, 224), (104, 117, 123))

# Diskten model yuklenmesi
print("Model yukleniyor ...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# ağa blob girişi ve tahmin alımı
net.setInput(blob)
baslangic = time.time()
tahminler = net.forward()
bitis = time.time()
print("Siniflandirma suresi >>> ",format(bitis - baslangic))

# en yüksek değere sahip 5 sınıflandırma sonucu
en_iyiler = np.argsort(tahminler[0])[::-1][:5]
print(tahminler.shape)

# en iyi tahminlerin yazdrılması
for (i, iyi) in enumerate(en_iyiler):
	if i == 0:
		yazi = "Sinif: {}, %{:.2f}".format(siniflar[iyi], tahminler[0][iyi] * 100)
		cv2.putText(resim, yazi, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	print("{}. Sinif: {}, >>  %{:.5}".format(i + 1, siniflar[iyi], tahminler[0][iyi] * 100))

#sonucu görelim
cv2.imshow("Resim", resim)
cv2.waitKey(0)