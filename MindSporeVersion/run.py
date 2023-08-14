from helper import *
from data_loader import *
import pickle
# sys.path.append('./')
from model.models import *
import random
from mindspore import Tensor
import mindspore as ms 
import mindspore.dataset as ds
from mindspore import nn ,ops
from mindspore.train import Model,LossMonitor, TimeMonitor,Metric
import mindspore.dataset as ds
from callback import Traincallback,Evalcallback

ms.set_context(device_target='GPU')
class MRRMetric(Metric):
	def __init__(self,):
		super(MRRMetric, self).__init__()
		self.results={"MRR":0.0,"MR":0.0,"count":0.0,'hits@1_count':0.0,'hits@3_count':0.0,'hits@10_count':0.0}

	def clear(self):
		"""Clears the internal evaluation result."""
		None

	def update(self,pred,label_obj):
		label,obj=label_obj
		b_range			= ops.arange(pred.shape[0]) #B
		target_pred		= pred[b_range, obj]

		pred 			= ops.where(label>0, -ops.ones_like(pred) * 10000000, pred) #所有标签不参与排序
		pred[b_range, obj] 	= target_pred
		ranks			= 1 + ops.argsort(ops.argsort(pred, axis=1, descending=True).astype(ms.float32), axis=1, descending=False)[b_range, obj]

		ranks 			= ranks.float()
		self.results['count']	= ops.numel(ranks) 		+ self.results.get('count', 0.0)
		self.results['MR']		= ops.sum(ranks)	+ self.results.get('MR',    0.0)
		self.results['MRR']		= ops.sum(1.0/ranks)   + self.results.get('MRR',   0.0)

		for k in [1,3,10]:
			self.results['hits@{}_count'.format(k)] = ops.numel(ranks[ranks <= (k)]) + self.results.get('hits@{}_count'.format(k), 0.0)

	def eval(self):

		mr = round(float(self.results["MR"] / self.results['count']),5)
		mrr =  round(float(self.results["MRR"] / self.results['count']),5)
		hit1 =  round(float(self.results["hits@1_count"] / self.results['count']),5)
		hit3 =  round(float(self.results["hits@3_count"] / self.results['count']),5)
		hit10 =  round(float(self.results["hits@10_count"] / self.results['count']),5)

		res = {"hits@1":hit1,"hits@3":hit3,"hits@10":hit10,"MRR":mrr,"MR":mr}
		print(res)
		return res	

class Runner(object):

	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

		if self.p.dataset == "FB15k-237":
			in_file = open('./data/{}/{}.pickle'.format(self.p.dataset, "desc_img_cls"),'rb')
			ent2img_desc = pickle.load(in_file)
		elif self.p.dataset == "WN9":
			in_file = open('./data/{}/{}.pickle'.format(self.p.dataset, "desc_img_cls"),'rb')
			ent2img_desc = pickle.load(in_file)
		else:
			raise "Current dataset does not implentment"
		in_file.close()
		ent2desc,ent2image={},{}
		for k,v in ent2img_desc.items():
			ent2desc[k] = v["desc"]
			ent2image[k] = v["img"]

		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.id2img =[]
		for i in range(len(self.id2ent.keys())):
			ent = self.id2ent[i]
			img = ent2image[ent]
			self.id2img.append(img)
		self.ent2img = Tensor(self.id2img)

		self.id2desc =[]
		for i in range(len(self.id2ent.keys())):
			ent = self.id2ent[i]
			desc = ent2desc[ent]
			self.id2desc.append(desc)
		self.ent2desc = Tensor(self.id2desc)

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim #默认embed_dim为None ，10*20

		self.data = ddict(list)
		sr2o = ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.data = dict(self.data) #{"train":[(s,r,o)] , "test":[],"valid":[]}

		self.sr2o = {k: list(v) for k, v in sr2o.items()} #{(s,r):[o1,o2......]} #训练集的映射
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()} #训练集、验证集、测试集的映射
		self.triples  = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}'.format(split)].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}'.format(split)].append({'triple': (obj, rel_inv, sub), 	'label': self.sr2o_all[(obj, rel_inv)]})

		'''
			self.triple ={
					"train":
					"test" :
					"valid":
			}
		'''
		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=False):
			column_names=["triples","label","mode"]
			dataset = dataset_class( self.triples[split], self.p )
			ms_dataset=ds.GeneratorDataset(source=dataset, column_names=column_names, shuffle= shuffle)
			ms_dataset = ms_dataset.batch(batch_size = batch_size)
			return  ms_dataset

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size, True),
			'valid':    	get_data_loader(TestDataset,  'valid', 		self.p.batch_size),
			'test':   		get_data_loader(TestDataset,  'test',  		self.p.batch_size),
		}

		self.train_num_steps = len(self.data_iter['train'])
		self.edge_index, self.edge_type = self.construct_adj()

	def construct_adj(self):
		"""
		Constructor of the runner class

		Parameters
		----------
		
		Returns
		-------
		Constructs the adjacency matrix for GCN
		
		"""
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= Tensor(edge_index).astype(ms.int32).t() #2*len(triples)
		edge_type	= Tensor(edge_type).astype(ms.int32)

		return edge_index, edge_type

	def __init__(self, params):
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer   = self.add_optimizer()

	def add_model(self, model, score_func):
		"""
		Creates the computational graph

		Parameters
		----------
		model_name:     Contains the model name to be created
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'mmkg_transe': 	model = MMKG_TransE(self.edge_index, self.edge_type, params=self.p ,ent2img=self.ent2img,ent2desc=self.ent2desc)
		elif model_name.lower()	== 'mmkg_distmult': 	model = MMKG_DistMult(self.edge_index, self.edge_type, params=self.p,ent2img=self.ent2img,ent2desc=self.ent2desc)
		elif model_name.lower()	== 'mmkg_conve': 	model = MMKG_ConvE(self.edge_index, self.edge_type, params=self.p,ent2img=self.ent2img,ent2desc=self.ent2desc)
		elif model_name.lower()	== 'mmkg_cp': 	model = MMKG_CP(self.edge_index, self.edge_type, params=self.p,ent2img=self.ent2img,ent2desc=self.ent2desc)
		else: raise NotImplementedError

		return model

	def add_optimizer(self):

		optimizer = nn.Adam(params=self.model.trainable_params(), learning_rate=self.p.lr,weight_decay=0.001)
		return optimizer

	def fit(self):
		ms_train_dataset=self.data_iter["train"]
		ms_valid_dataset=self.data_iter["valid"]
		ms_test_dataset=self.data_iter["test"]

		model = Model(self.model, optimizer=self.optimizer,eval_network=self.model,eval_indexes =[0,1,2],metrics={"MRRMetric":MRRMetric()})

		hist = {'loss':[], 'MRR':[], 'hits@1':[], 'hits@3':[], 'hits@10':[]} # 训练过程记录
		train_cb=Traincallback(hist["loss"])
		eval_cb= Evalcallback(self.p,model,hist,ms_valid_dataset,ms_test_dataset)
		model.train(self.p.max_epochs,ms_train_dataset,callbacks=[train_cb,eval_cb,LossMonitor(100),TimeMonitor(0)])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='mmkg',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='mult')

	parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN') #150
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use') #2
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	parser.add_argument('-rnn_layers',	 default=1,   	type=int, 	help='rnn_layers')
	parser.add_argument('-rnn_model',	 default="gru",   	type=str, 	help='rnn_model')

	args = parser.parse_args()

	if not args.restore: 
		args.name = args.dataset + '_' +args.score_func + '_' + args.opn + '_'+  str(args.rnn_layers) + '_'+args.rnn_model + '_'+time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')


	model = Runner(args)
	model.fit()
