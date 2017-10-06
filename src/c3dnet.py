#! bin/usr/python
# -*- coding:utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import cupy as np

class C3D(chainer.Chain):

    """Convolution3Dnet"""

    #insize = 227

    def __init__(self):
        super(C3D, self).__init__(
            c1_1=L.ConvolutionND(3,3,32,ksize = 3, stride=2, pad = 0),
            
            c2_1=L.ConvolutionND(3,32,64,ksize=3, stride=1, pad = 0),
            
            c3_1=L.ConvolutionND(3,64,128,ksize=3, stride=1, pad = 0),
            #c3_2=L.ConvolutionND(3,256,256,ksize=3, stride=1, pad = 0),
            
            #c4_1=L.ConvolutionND(3,None,512,ksize=(3,3,1), stride=1, pad = 0),
            #c4_2=L.ConvolutionND(3,512,512,ksize=3, stride=1, pad = 0),
            
            #c5_1=L.ConvolutionND(3,512,512,ksize=(1,1,1), stride=1, pad = 0),
            #c5_2=L.ConvolutionND(3,512,512,ksize=3, stride=1, pad = 0),
            
            b1_1=L.BatchNormalization(32),
            b2_1=L.BatchNormalization(64),
            b3_1=L.BatchNormalization(128),
            #b3_2=L.BatchNormalization(256),
            #b4_1=L.BatchNormalization(512),
            #b4_2=L.BatchNormalization(512),
            #b5_1=L.BatchNormalization(512),
            #b5_2=L.BatchNormalization(512),
            
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 3)
        )
        self.train = True

    def __call__(self, x, t):
        print ('call')
        h = self.b1_1(F.relu(self.c1_1(x)))
        h = self.b2_1(F.relu(self.c2_1(h)))
        h = self.b3_1(F.relu(self.c3_1(h)))
        #h = self.b3_2(F.relu(self.c3_2(h)))
        #h = self.b4_1(F.relu(self.c4_1(h)))
        #h = self.b4_2(F.relu(self.c4_2(h)))
        #h = self.b5_1(F.relu(self.c5_1(h)))
        #h = self.b5_2(F.relu(self.c5_2(h)))
        h = self.fc6(h)
        h = self.fc7(h)
 
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class LSTM_VGG(chainer.Chain):
    """VGGにフレームを与えてLSTMにいれる"""
    """VGGNetbase
    - It takes (224, 224, 3) sized image as imput
    """

    def __init__(self):
        super(LSTM_VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            #conv3_1_b=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            #conv3_2_b=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            #conv3_3_b=L.Convolution2D(None, 512, 3, stride=1, pad=1),

            conv4_1_b=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            #conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            #conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            #fc6_b=L.Convolution2D(128, 128, 7),
            #fc7=L.Convolution2D(4096, 4096, 1),
            #fc8=L.Convolution2D(4096, 1, 1)
            #fc9=L.Linear(4096, 4)
            #ここからLSTM
            l8=L.LSTM(None, 128),
            fc9=L.Linear(None, 3)
        )
        self.train = True

    def __call__(self, *x_list):
        #print (x_list.data.shape)
        #print (x_list.data[0])
        #print (x_list.data[1])
        ##１フレームごとfc7までを検出しLSTMに対してループ
        self.l8.reset_state()
        
        y_list=[]
        for i in range(5):
            x = x_list[i]
            #x.volatile = 'on' 
            #print (xd.shape) 
            #xd = xd.asarray(xd).astype("f")
            #x = np.asarray(xd, dtype=np.float32)
            #print (x.data.shape)
            #print (x.shape)
            #x = np.asarray(xd).astype(np.float32)
            
            #print (i)
            h = F.dropout(F.relu(self.conv1_1(x)))
            h = F.dropout(F.relu(self.conv1_2(h)))
            h = F.max_pooling_2d(h, 2, stride=2)

            h = F.dropout(F.relu(self.conv2_1(h)))
            h = F.dropout(F.relu(self.conv2_2(h)))
            h = F.max_pooling_2d(h, 2, stride=1)

            #h = F.dropout(F.relu(self.conv3_1_b(h)))
            #h = F.relu(self.conv3_2_b(h))
            #h = F.relu(self.conv3_3_b(h))
            #h = F.max_pooling_2d(h, 2, stride=2)

            h = F.dropout(F.relu(self.conv4_1_b(h)))
            #h = F.relu(self.conv4_2(h))
            #h = F.relu(self.conv4_3(h))
            #h = F.max_pooling_2d(h, 2, stride=2)

            #h = F.relu(self.conv5_1(h))
            #h = F.relu(self.conv5_2(h))
            #h = F.relu(self.conv5_3(h))
            #h = F.max_pooling_2d(h, 2, stride=2)

            #h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
            #h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
            
            #h.volatile = 'off'
            #h = F.relu(self.fc6_b(h))
            #h = F.relu(self.fc7(h))
             
            l = self.l8(h)
            
            #print h.data
            #h = F.sigmoid(self.fc9(h))
            #h = (h*4)+1
            #print (h.data)
        h = self.fc9(l)
        
        if self.train:
            #self.loss = F.softmax_cross_entropy(h, t)
            #chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            #print ("return loss")
            return h 
            #return self.loss
            
        else:
            self.pred = F.softmax(l)
            return self.pred

    def reset_state(self):
        self.l8.reset_state()
