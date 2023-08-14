from helper import *
from model.message_passing import MessagePassing,MessagePassingimg
import torch 
from typing import Any, Optional, Tuple
from torch import nn 
class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels #100
		self.out_channels	= out_channels #150
		self.num_rels 		= num_rels
		self.act 		= act #激活函数
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels)) #100,150
		self.w_in		= get_param((in_channels, out_channels)) #100,150
		self.w_out		= get_param((in_channels, out_channels)) #100,150

		self.w_loop_img		= get_param((in_channels, out_channels)) #100,150
		self.w_in_img		= get_param((in_channels, out_channels)) #100,150
		self.w_out_img		= get_param((in_channels, out_channels)) #100,150

		self.w_loop_desc	= get_param((in_channels, out_channels)) #100,150
		self.w_in_desc		= get_param((in_channels, out_channels)) #100,150
		self.w_out_desc		= get_param((in_channels, out_channels)) #100,150

		self.w_rel 		= get_param((in_channels, out_channels)) #100,150
		self.loop_rel 	= get_param((1, in_channels));       #1,100
		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		# self.fusion 		= get_param((in_channels*2, out_channels)) #100,150
		# print("#"*50,in_channels,out_channels)
		if self.p.bias: #默认为False
			self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

		self.modality_weight = torch.nn.Parameter(torch.Tensor(self.p.num_ent,3))
		if self.p.rnn_model=="gru":
			self.rnn = nn.GRU(input_size = in_channels, hidden_size = in_channels//2, num_layers=self.p.rnn_layers, bidirectional = True)
		elif self.p.rnn_model=="lstm":
			self.rnn = nn.LSTM(input_size = in_channels, hidden_size = in_channels//2, num_layers=self.p.rnn_layers, bidirectional = True)
		else:
			raise "NotImplement" 
		self.drop_fusion	= torch.nn.Dropout(self.p.dropout)
		for n,v in self.rnn.named_parameters():
			if "weight" in n :
				v.data = get_param(v.data.shape)
			elif "bias" in n :
				v.data = torch.zeros_like(v.data)


	def forward(self, x, edge_index, edge_type, rel_embed, ent2img,ent2desc): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2  #反向边
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:] #[2,num_edges] [2,num_edges] 实体下标
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:] #[2,num_edges] [2,num_edges] 关系下标

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		ent_embed = torch.cat((x.unsqueeze(0),ent2img.unsqueeze(0),ent2desc.unsqueeze(0)),0)  #2,edge_num,hidden
		output,ent_embed = self.rnn(input = ent_embed)
		x		= output[0]
		ent2img	= output[1]
		ent2desc =output[2]

		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in' ,ent2img=ent2img,ent2desc=ent2desc,modality_weight=self.modality_weight)
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop',ent2img=ent2img,ent2desc=ent2desc,modality_weight=self.modality_weight)
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out',ent2img=ent2img,ent2desc=ent2desc,modality_weight=self.modality_weight)
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def rel_transform2(self, ent_embed, rel_embed,ent_embed_img,ent_embed_desc,modality_weight):
		if   self.p.opn == 'corr': 	
			trans_embed  		= ccorr(ent_embed, 	rel_embed)
			trans_embed_img  	= ccorr(ent_embed_img, rel_embed)
			trans_embed_desc  	= ccorr(ent_embed_desc, 	rel_embed)

		elif self.p.opn == 'sub': 	
			trans_embed  		= ent_embed  		- rel_embed
			trans_embed_img  	= ent_embed_img  	- rel_embed
			trans_embed_desc  	= ent_embed_desc  	- rel_embed

		elif self.p.opn == 'mult': 	
			trans_embed  		= ent_embed  		* rel_embed
			trans_embed_img  	= ent_embed_img  	* rel_embed
			trans_embed_desc  	= ent_embed_desc  	* rel_embed

		else: 
			raise NotImplementedError

		return trans_embed,trans_embed_img,trans_embed_desc


	def message(self, x_j, edge_type, rel_embed, edge_norm, mode,x_j_img,x_j_desc,modality_weight):
		weight 		= getattr(self, 'w_{}'.format(mode))
		weight_img 	= getattr(self, 'w_{}_img'.format(mode))
		weight_desc = getattr(self, 'w_{}_desc'.format(mode))

		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel,xj_rel_img,xj_rel_desc  = self.rel_transform2(x_j, rel_emb,x_j_img,x_j_desc,modality_weight)
		# xj_rel  = self.rel_transform(x_j, rel_emb)

		out			= torch.mm(xj_rel, weight)
		out_img		= torch.mm(xj_rel_img, weight_img)
		out_desc	= torch.mm(xj_rel_desc, weight_desc)

		out = self.drop_fusion(out)*(1/3) + self.drop_fusion(out_img)*(1/3) + self.drop_fusion(out_desc)*(1/3)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class CompGCNConvimg(MessagePassingimg):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels #100
		self.out_channels	= out_channels #150
		self.num_rels 		= num_rels
		self.act 		= act #激活函数
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels)) #100,150
		self.w_in		= get_param((in_channels, out_channels)) #100,150
		self.w_out		= get_param((in_channels, out_channels)) #100,150
		self.w_rel 		= get_param((in_channels, out_channels)) #100,150
		self.loop_rel 		= get_param((1, in_channels));       #1,100
		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		# self.fusion 		= get_param((in_channels*2, out_channels)) #100,150
		# print("#"*50,in_channels,out_channels)
		if self.p.bias: #默认为False
			self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2  #反向边
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:] #[2,num_edges] [2,num_edges]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:] #[2,num_edges] [2,num_edges]

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in' )
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel  = self.rel_transform(x_j, rel_emb)

		out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

