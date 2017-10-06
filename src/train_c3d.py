#!/usr/bin/env python
#coding: UTF-8

from __future__ import print_function

import chainer
import numpy as np
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Variable, serializers, training
from chainer.training import extensions
from chainer.links import CaffeFunction

import videoread
import c3dnet
import copy_model

import argparse


"""
バイナリから呼び出す時の記述 //たぶん書き換えないと動きません
data = np.empty((0,18,240,320,3), dtype=np.float32)
for file in filelist:
	print  file
	with open(file, mode ='r') as f:
		single_data = pickle.load(f)
		data = np.append(data, single_data["data"])
"""

def main():
	#オプションの追加
	parser = argparse.ArgumentParser(description='Chainer : C3D')
	parser.add_argument('--arch', '-a', default='ADAM',
                        help='Convnet architecture')
	parser.add_argument('--batchsize', '-b', type=int, default=1,
	                    help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=100,
	                    help='Number of sweeps over the dataset to train')
	parser.add_argument('--gpu', '-g', type=int, default=0,
	                    help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='../result/64x64_20/20170913',
	                    help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='resume',
	                    help='Resume the training from snapshot')
	parser.add_argument('--unit', '-u', type=int, default=1000,
	                    help='Number of units')
	parser.add_argument('--input', '-i', default='../data/classify_4_64x64',
	                    help='Directory to input data')

	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# unit: {}'.format(args.unit))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('# input: {}'.format(args.input))
	print('')


        #データセットの読み込み 
        v = videoread.VideoRead()
	train, test = v.combine_data_label(args.input)
	
	#学習済みCaffeモデルの設定
        pre_model = CaffeFunction("c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel")
        print (pre_model.shape)
        
        model = c3dnet.C3D()
	if args.gpu >= 0:
		chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
		model.to_gpu()  # Copy the model to the GPU

	#Setup an optimizer"
	optimizer = chainer.optimizers.MomentumSGD()
	optimizer.setup(model)

	#make iterators"
	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
	#set up a trainer"
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


	val_interval = (20 ), 'iteration'
	log_interval = (20 ), 'iteration'
	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
	trainer.extend(extensions.dump_graph('main/loss'))
	trainer.extend(extensions.snapshot(), trigger=val_interval)
	trainer.extend(extensions.snapshot_object(
	    model, 'model_iter_{.updater.iteration}'), trigger=val_interval)


	trainer.extend(extensions.LogReport(trigger=log_interval))
	trainer.extend(extensions.observe_lr(), trigger=log_interval)
	trainer.extend(extensions.PrintReport([
	    'epoch', 'iteration', 'main/loss', 'validation/main/loss',
	    'main/accuracy', 'validation/main/accuracy', 'lr'
	]), trigger=log_interval)
	#Progress barを表示
	trainer.extend(extensions.ProgressBar())#update_interval=10))

        #trainer.run()
        serializers.save_npz("mymodel.npz", model)
	serializers.save_npz("mynet.npz", net)


if __name__ == '__main__':
	main()
