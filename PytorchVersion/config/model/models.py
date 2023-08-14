from helper import *
from model.compgcn_conv import CompGCNConv,CompGCNConvimg
from model.compgcn_conv_basis import CompGCNConvBasis
from torch import nn 
# from att import CLIPEncoderLayer

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class MMKGBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None, ent2img=None,ent2desc=None):
		super(MMKGBase, self).__init__(params)
		assert ent2img is not None
		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim   #150
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim)) #[ent,100]
		self.device		= self.edge_index.device
		
		# print(self.init_embed.shape)
		# exit()
		self.ent2img = ent2img.to(torch.float).to(self.device)
		self.img_trans = nn.Parameter(torch.Tensor(768, self.p.init_dim), requires_grad=True).to(self.device)
		nn.init.xavier_uniform_(self.img_trans)  # 每层网络保证输入输出的方差相同

		self.ent2desc = ent2desc.to(torch.float).to(self.device)
		self.desc_trans = nn.Parameter(torch.Tensor(768, self.p.init_dim), requires_grad=True).to(self.device)
		nn.init.xavier_uniform_(self.desc_trans)  # 每层网络保证输入输出的方差相同

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	
				self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					
				self.init_rel = get_param((num_rel*2, self.p.init_dim)) #[2 * rel ,100]

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		
		ent2img = torch.mm(self.ent2img,self.img_trans) 
		ent2desc = torch.mm(self.ent2desc,self.desc_trans) 

		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r, ent2img=ent2img, ent2desc=ent2desc)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r,ent2img=ent2img,ent2desc=ent2desc) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x


		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class MMKG_TransE(MMKGBase):
	def __init__(self, edge_index, edge_type, params=None,ent2img=None,ent2desc=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, ent2img,ent2desc)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x) #[B,ent_num] FB15k-237-->[B,14541]
		# print(score.shape)
		return score

class MMKG_DistMult(MMKGBase):
	def __init__(self, edge_index, edge_type, params=None,ent2img=None,ent2desc=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params,ent2img,ent2desc)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		print("Here is CompGCN_DistMult")
	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class MMKG_ConvE(MMKGBase):
	def __init__(self, edge_index, edge_type, params=None,ent2img=None,ent2desc=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, ent2img,ent2desc)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)
		score = torch.sigmoid(x)
		return score

class MMKG_CP(MMKGBase):
	def __init__(self, edge_index, edge_type, params=None,ent2img=None,ent2desc=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, ent2img,ent2desc)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		dim = sub_emb.shape[1] // 2

		sub = sub_emb[:,:dim],sub_emb[:,dim:]
		rel = rel_emb[:,:dim],rel_emb[:,dim:]
		all_ent = all_ent[:,:dim],all_ent[:,dim:]

		x = (sub[0] * rel[0] - sub[1] * rel[1]) @ all_ent[0].transpose(0, 1) + \
			(sub[0] * rel[1] + sub[1] * rel[0]) @ all_ent[1].transpose(0, 1)

		# obj_emb				= sub_emb + rel_emb
		x += self.bias.expand_as(x)

		score	= torch.sigmoid(x) #[B,ent_num] FB15k-237-->[B,14541]
		# print(score.shape)
		return score
