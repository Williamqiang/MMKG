import inspect
from mindspore import nn ,ops  
from mindspore import Tensor, Parameter
import numpy as np
def scatter_(name, src, index, dim_size=None):
	index = index.unsqueeze(-1)
	index = index.repeat(src.shape[-1],-1)
	input_x = Parameter(Tensor(np.zeros((dim_size, 200))).astype(np.float32), name=None)
	out = ops.tensor_scatter_elements(input_x,indices=index,updates=src,axis=0,reduction="add")
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(nn.Cell):
	def __init__(self, aggr='add'):
		super(MessagePassing, self).__init__()
		
		#https://blog.csdn.net/weixin_45501561/article/details/112827975
		#FullArgSpec(args=[], varargs=None, varkw=None, defaults=('123',), kwonlyargs=[], kwonlydefaults=None, annotations={})

		#x_j, edge_type, rel_embed, edge_norm, mode
		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out
		# self.weight = None
	def propagate(self, aggr, edge_index, **kwargs):
		kwargs['edge_index'] = edge_index

		size = None
		message_args = []
		img_value=None
		for arg in self.message_args:
			##x_j,edge_type, rel_embed, edge_norm, mode ,x_j_img
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x', 
				size = tmp.shape[0]
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
				img_value = kwargs["ent2img"][edge_index[0]]
				desc_value = kwargs["ent2desc"][edge_index[0]]
			elif arg[-2:] == '_j':						# Take things from kwargs ###############################默认这一个 
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.shape[0]
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
				img_value = kwargs["ent2img"][edge_index[1]]
				desc_value = kwargs["ent2desc"][edge_index[1]]

			elif arg=='x_j_img':
				message_args.append(img_value)

			elif arg=='x_j_desc':
				message_args.append(desc_value)

			else:
				message_args.append(kwargs[arg])		

		# message_args.append(img_value)
		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		return x_j

	def update(self, aggr_out):  # pragma: no cover
		return aggr_out


