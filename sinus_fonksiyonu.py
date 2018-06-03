#https://www.symbolengine.com/sunidimag/2017/08/04/101-Ilk-yapay-sinir-agimiz-Dense-layer/
from keras.layers import Dense
from keras.models import Sequential
import random
import matplotlib.pyplot as plt
import math

################################################ Eğitim Verisi Oluşturuldu Ve Çizdirildi
gurultu = 0.2
X = []
y = []
for i in range(0,1000):
    angle=random.uniform(-math.pi,math.pi)
    X.append(angle)
    y.append(math.sin(angle)+random.uniform(-gurultu,gurultu))

plt.scatter(X,y,s=0.1)
plt.xlabel('x (Radyan)')
plt.ylabel('sin(x)')
plt.legend()
plt.show()

################################################ Model Tanımlanıyor
model = Sequential()
model.add(Dense(100, input_shape=(1,), activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='sgd')

################################################ Model Eğitimi
history=model.fit(X, y, epochs=500, verbose=1)

################################ loss değerinin değişimini çizdiriyoruz
print(history.history.keys())
plt.plot(history.history['loss'])
plt.grid()
plt.show()

######################### Test Verisi Oluşturuldu
X_test = []
y_test = []
for i in range(-1800,1800):
    angle = math.radians(i/10)
    X_test.append(angle)
    y_test.append(math.sin(angle))

########################## Modeli test ederek gerçek ve tahmin edilen değerler gösteriliyor
def testmodel(X,y):
    res = model.predict(X, batch_size=256)
    plt.plot(X,y, label='sin')
    plt.plot(X,res, label='sonuc')
    plt.xlabel('x (Radyan)')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.show()

testmodel(X_test,y_test)

###########################Kullanıcının modeli test etmesi için arayüz
girilen=float(input("Radyan Değerini Griniz= "))
y_proba = model.predict([girilen])
print("Gerçek Sinüs Değeri= ",math.sin(girilen))
print("Makinenin Öğrendiği Sinüs Değeri= ",y_proba)