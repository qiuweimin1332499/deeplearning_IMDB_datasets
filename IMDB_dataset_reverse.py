from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

max_features=10000
maxlen=100
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=[x[::-1] for x in x_train]
x_test=[x[::-1] for x in x_test]
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)
#序列反转
model=Sequential()
model.add(layers.Embedding(max_features,128))
model.add(layers.LSTM(32,dropout=0.2,recurrent_dropout=0.2))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history=model.fit(x_train,
                  y_train,
                  epochs=10,
                  batch_size=32,
                  validation_split=0.2)
#建立模型
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
epochs=range(1,len(acc)+1)
plt.figure()
plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('training and validation acc')
plt.legend()
plt.show()
#显示结果