#!/usr/bin/env python
#coding: UTF-8
import cv2
import os
import numpy as np
from chainer import cuda, datasets
xp = np

cp = cuda.cupy #ここをnumpyかcupyに
import six.moves.cPickle as pickle

"""
chainerでは配列の形を[id番号, チャネル数，d1, d2, d3]
にする必要がある
im00くらいでしか動かないのでそこで配列のファイルを生成
してください．
→gpマシンで動くようになりました．
バイナリに書き込まなくても問題ありません．
"""

class VideoRead:

	def makelist(self, filename): #1つの動画に対する処理
            assert(os.path.exists(filename)), "not exist file"
            video_path = filename #"/export/data/dataset/UCF-101/Archery/v_Archery_g01_c01.avi"
            framenum = 0
            name, ext = os.path.splitext(filename)
            cap = cv2.VideoCapture(video_path)
            videolist = xp.empty((0,128,171,3), dtype=xp.float32)

            while(framenum < 280):
                if(framenum % 35 == 0):
                    ret, frame = cap.read()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    try:
                        frame = xp.array(frame, dtype=xp.float32)
                        frame = frame[xp.newaxis]
                        videolist = xp.append(videolist, frame, axis=0)
                    except:
                        print("break")
                        break
                framenum += 1
            cap.release()

            videolist = videolist.reshape((1,) + videolist.shape)
            print (videolist.shape)
            return videolist


	def makelist_dir(self, dirname, num_label): #1つのカテゴリに対する処理
            assert(os.path.exists(dirname)), "not exist directory"
            filelist = os.listdir(dirname)
            all_video = xp.empty((0,8,128,171,3), dtype=xp.float32)

            for file in filelist:
                all_video = xp.append(all_video, self.makelist(dirname+"/"+file), axis=0)
            all_video = all_video.transpose(0,4,1,2,3)
            
            #------バイナリに書き込む用の記述--------------------------------
            #with open(dirname+'.pkl', mode ='wb') as f:
                    #assert(os.path.exists('binary_list')), "not exist directory"
            #	pickle.dump(data,f)
            #----------------------------------------------------------------
            label_list = xp.ones(len(filelist), dtype=xp.int32) * num_label
            
            return all_video, label_list


	
	def makelist_all_class(self, path): #各カテゴリごとにデータを生成
            num_label = 1 #ラベル番号
            assert(os.path.exists(path)), "not exist directory"
            dirlist = os.listdir(path)
            data = xp.empty((0,3,8,128,171), dtype=xp.float32)
            labels = xp.empty((0), dtype=xp.int32)
            for dir in dirlist:
                print dir
                datum, label = self.makelist_dir(os.path.join(path, dir), num_label)
                data = xp.append(data, datum, axis=0)
                labels = xp.append(labels, label, axis=0)
                num_label += 1

            return data, labels

	def combine_data_label(self, path): #TupleDataset型に変換
            assert(os.path.exists(path)), "not exist directory"
            data, label = self.makelist_all_class(path)
            N = 1 #学習に使用するサイズ
            #学習用とテスト用にファイルを分ける	
            with cuda.Device(1):
                x_gpu = cp.array(data)
                y_gpu = cp.array(label)
                x_train, x_test = cp.split(x_gpu, [N])
                y_train, y_test = cp.split(y_gpu, [N])
            #1.11以降用のデータセット定義: TupleDataset型
            train = datasets.tuple_dataset.TupleDataset(x_train, y_train)
            test = datasets.tuple_dataset.TupleDataset(x_test, y_test)

            print type(x_train)
            return train, test

