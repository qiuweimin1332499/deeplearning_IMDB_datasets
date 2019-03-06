from keras import backend as K
from keras.engine.topology import Layer

class Position_Embedding(Layer):
    def __init__(self, size=None, **kwargs):
        self.size = size  # 必须为偶数
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        # K.arange不支持数据类型及长度，只好用这种方法生成，不过很奇怪为什么这样就能生成浮点类型的数据
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        return position_ij + x

    def compute_output_shape(self, input_shape):
            return input_shape



from keras.engine.topology import Layer

class Attention(Layer):

    def __init__(self,nb_head,size_per_head,**kwargs):
        self.nb_head=nb_head
        self.size_per_head=size_per_head
        self.output_dim=nb_head*size_per_head
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.WQ=self.add_weight(name='WQ',
                                shape=(input_shape[0][-1], self.output_dim),
                                initializer='glorot_uniform',
                                trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention,self).build(input_shape)

    def call(self,x):
        Q_seq,K_seq,V_seq=x
        Q_seq=K.dot(Q_seq,self.WQ)
        Q_seq=K.reshape(Q_seq,(-1,K.shape(Q_seq)[1],self.nb_head,self.size_per_head))
        Q_seq=K.permute_dimensions(Q_seq,(0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        O_seq=K.batch_dot(Q_seq,K_seq,axes=[3,3])/self.size_per_head**0.5
        O_seq=K.softmax(O_seq)
        O_seq=K.batch_dot(O_seq,V_seq,axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq=K.reshape(O_seq,(-1,K.shape(O_seq)[1],self.output_dim))
        return O_seq

    def compute_output_shape(self, input_shape):
        return(input_shape[0][0],input_shape[0][1],self.output_dim)




from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import *
from keras.models import Model
max_features=10000
maxlen=100
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)
#序列填充
S_inputs=Input(shape=(None,),dtype='int32')
embeddings=Embedding(max_features,128)(S_inputs)
#embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq=Attention(8,16)([embeddings,embeddings,embeddings])#三输入多头注意力模型
O_seq=GlobalAveragePooling1D()(O_seq)
O_seq=Dropout(0.2)(O_seq)
outputs=Dense(1,activation='sigmoid')(O_seq)
model=Model(inputs=S_inputs,outputs=outputs)
#网络结构
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



