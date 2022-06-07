import os
import argparse

def get_args():
	parser = argparse.ArgumentParser(description = "English Past Tense")

	# random seed settings
	parser.add_argument()

	# data path
	parser.add_argument('-data_path_train', type = str, default = 'https://raw.githubusercontent.com/xiaomeng-ma/English-Past-Tense/master/Data/Training/Train_type_reg.csv?token=GHSAT0AAAAAABUKNZ7WDYTWMTYVVWUWJJJUYU7X4ZA', help = 'training data path')
	parser.add_argument('-data_path_test', type = str, default='https://raw.githubusercontent.com/xiaomeng-ma/English-Past-Tense/master/Data/Test/Test.csv?token=GHSAT0AAAAAABUKNZ7WOAQNKH54KXWTWOL4YU7X7UQ', help = 'test data path')
	parser.add_argument('-data_path_nonce', type = str, default='https://raw.githubusercontent.com/xiaomeng-ma/English-Past-Tense/master/Data/Test/Nonce.csv?token=GHSAT0AAAAAABUKNZ7XPQSSVOVHF4WQ7BXSYU7YAOQ', help = 'nonce data path')

	#label
	parser.add_argument('-add_label', action = 'store_true', help = 'add label')
	parser.add_argument('-label_spec', type = str, default = 'reg', choices = ['reg', 'vc', 'both'], help = 'reg: regularity, vc: verb class, both: both')

	return parser.parse_args()
