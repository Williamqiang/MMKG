from helper import *

class TrainDataset():
	"""
	Training Dataset class.

	Parameters
	----------
	triples:	The triples used for training the model
	params:		Parameters for the experiments
	
	Returns
	-------
	A training Dataset class instance
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= np.int32(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label,np.array(0)

	# @staticmethod
	# def collate_fn(data):
	# 	triple		= np.stack([_[0] 	for _ in data], dim=0)
	# 	trp_label	= np.stack([_[1] 	for _ in data], dim=0)
	# 	return triple, trp_label
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			pos_obj		= label
			mask		= np.ones([self.p.num_ent], dtype=np.bool)
			mask[label]	= 0
			neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
			neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

			return neg_ent

		neg_ent = get(triple, label)
		return neg_ent

	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return y


class TestDataset():
	"""
	Evaluation Dataset class.

	Parameters
	----------
	triples:	The triples used for evaluating the model
	params:		Parameters for the experiments
	
	Returns
	-------
	An evaluation Dataset class instance used by DataLoader for model evaluation
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= np.int32(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label,np.array(1)

	# @staticmethod
	# def collate_fn(data):
	# 	triple		= np.stack([_[0] 	for _ in data], dim=0)
	# 	label		= np.stack([_[1] 	for _ in data], dim=0)
	# 	return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return y