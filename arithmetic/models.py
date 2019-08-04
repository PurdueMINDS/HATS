import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models import inception_v3
import pdb
import torch.nn.functional as F
from torch.autograd import Variable
xrange = range

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()



def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()



def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)



class IntraRNN(nn.Module):
    
    
    def __init__(self, batch_size, num_tokens, embed_size, intra_gru_hidden, bidirectional= True):        
        
        super(IntraRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.intra_gru_hidden = intra_gru_hidden
        #self.bidirectional = bidirectional
        bidirectional = False
        self.bidirectional = False
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            self.intra_gru = nn.GRU(embed_size, intra_gru_hidden, bidirectional= True)
            self.weight_W_intra = nn.Parameter(torch.Tensor(2* intra_gru_hidden,2*intra_gru_hidden))
            self.bias_intra = nn.Parameter(torch.Tensor(2* intra_gru_hidden,1))
            self.weight_proj_intra = nn.Parameter(torch.Tensor(2*intra_gru_hidden, 1))
        else:
            self.intra_gru = nn.GRU(embed_size, intra_gru_hidden, bidirectional= False)
            self.weight_W_intra = nn.Parameter(torch.Tensor(intra_gru_hidden, intra_gru_hidden))
            self.bias_intra = nn.Parameter(torch.Tensor(intra_gru_hidden,1))
            self.weight_proj_intra = nn.Parameter(torch.Tensor(intra_gru_hidden, 1))
            
        self.softmax_intra = nn.Softmax()
        self.weight_W_intra.data.uniform_(-0.1, 0.1)
        self.weight_proj_intra.data.uniform_(-0.1,0.1)
        self.bias_intra.data.uniform_(-0.1,0.1)

        
        
    def forward(self, embed, state_intra):
        embedded = self.lookup(embed)
        output_intra, state_intra = self.intra_gru(embedded, state_intra)
        return output_intra[-1,:,:].squeeze(), state_intra #, intra_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.intra_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.intra_gru_hidden))   

# ## Intra attention model with bias

class AttentionIntraRNN(nn.Module):
    
    
    def __init__(self, batch_size, num_tokens, embed_size, intra_gru_hidden, bidirectional= True):        
        
        super(AttentionIntraRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.intra_gru_hidden = intra_gru_hidden
        self.bidirectional = bidirectional
        
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            self.intra_gru = nn.GRU(embed_size, intra_gru_hidden, bidirectional= True)
            self.weight_W_intra = nn.Parameter(torch.Tensor(2* intra_gru_hidden,2*intra_gru_hidden))
            self.bias_intra = nn.Parameter(torch.Tensor(2* intra_gru_hidden,1))
            self.weight_proj_intra = nn.Parameter(torch.Tensor(2*intra_gru_hidden, 1))
        else:
            self.intra_gru = nn.GRU(embed_size, intra_gru_hidden, bidirectional= False)
            self.weight_W_intra = nn.Parameter(torch.Tensor(intra_gru_hidden, intra_gru_hidden))
            self.bias_intra = nn.Parameter(torch.Tensor(intra_gru_hidden,1))
            self.weight_proj_intra = nn.Parameter(torch.Tensor(intra_gru_hidden, 1))
            
        self.softmax_intra = nn.Softmax()
        self.weight_W_intra.data.uniform_(-0.1, 0.1)
        self.weight_proj_intra.data.uniform_(-0.1,0.1)
        self.bias_intra.data.uniform_(-0.1,0.1)

        
        
    def forward(self, embed, state_intra):
        # embeddings
        #pdb.set_trace()
        embedded = self.lookup(embed)
        # intra level gru
        output_intra, state_intra = self.intra_gru(embedded, state_intra)
#         print output_intra.size()
        intra_squish = batch_matmul_bias(output_intra, self.weight_W_intra,self.bias_intra, nonlinearity='tanh')
        intra_attn = batch_matmul(intra_squish, self.weight_proj_intra)
        intra_attn_norm = self.softmax_intra(intra_attn.transpose(1,0))
        intra_attn_vectors = attention_mul(output_intra, intra_attn_norm.transpose(1,0))        
        return intra_attn_vectors , state_intra, intra_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.intra_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.intra_gru_hidden))        


# ## Interset Attention model with bias


class InterRNN(nn.Module):
    
    
    def __init__(self, batch_size, inter_gru_hidden, intra_gru_hidden, n_classes, bidirectional= True):        
        
        super(InterRNN, self).__init__()
        
        self.batch_size = batch_size
        self.inter_gru_hidden = inter_gru_hidden
        self.n_classes = n_classes
        self.intra_gru_hidden = intra_gru_hidden
        bidirectional = False
        self.bidirectional = bidirectional
        
        
        if bidirectional == True:
            self.inter_gru = nn.GRU(2 * intra_gru_hidden, inter_gru_hidden, bidirectional= True)        
            self.weight_W_inter = nn.Parameter(torch.Tensor(2* inter_gru_hidden ,2* inter_gru_hidden))
            self.bias_inter = nn.Parameter(torch.Tensor(2* inter_gru_hidden,1))
            self.weight_proj_inter = nn.Parameter(torch.Tensor(2* inter_gru_hidden, 1))
            self.final_linear = nn.Linear(2* inter_gru_hidden, n_classes)
        else:
            self.inter_gru = nn.GRU(intra_gru_hidden, inter_gru_hidden, bidirectional= False)        
            self.weight_W_inter = nn.Parameter(torch.Tensor(inter_gru_hidden ,inter_gru_hidden))
            self.bias_inter = nn.Parameter(torch.Tensor(inter_gru_hidden,1))
            self.weight_proj_inter = nn.Parameter(torch.Tensor(inter_gru_hidden, 1))
            self.final_linear = nn.Linear(inter_gru_hidden, n_classes)
        self.softmax_inter = nn.Softmax()
        self.final_softmax = nn.Softmax()
        self.weight_W_inter.data.uniform_(-0.1, 0.1)
        self.weight_proj_inter.data.uniform_(-0.1,0.1)
        self.bias_inter.data.uniform_(-0.1,0.1)
        
        
    def forward(self, intra_attention_vectors, state_inter):
        #pdb.set_trace()
        output_inter, state_inter = self.inter_gru(intra_attention_vectors, state_inter)        
        return output_inter[-1,:,:].squeeze()
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.inter_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.inter_gru_hidden))

class AttentionInterRNN(nn.Module):
    
    
    def __init__(self, batch_size, inter_gru_hidden, intra_gru_hidden, n_classes, bidirectional= True):        
        
        super(AttentionInterRNN, self).__init__()
        
        self.batch_size = batch_size
        self.inter_gru_hidden = inter_gru_hidden
        self.n_classes = n_classes
        self.intra_gru_hidden = intra_gru_hidden
        self.bidirectional = bidirectional
        
        
        if bidirectional == True:
            self.inter_gru = nn.GRU(2 * intra_gru_hidden, inter_gru_hidden, bidirectional= True)        
            self.weight_W_inter = nn.Parameter(torch.Tensor(2* inter_gru_hidden ,2* inter_gru_hidden))
            self.bias_inter = nn.Parameter(torch.Tensor(2* inter_gru_hidden,1))
            self.weight_proj_inter = nn.Parameter(torch.Tensor(2* inter_gru_hidden, 1))
            self.final_linear = nn.Linear(2* inter_gru_hidden, n_classes)
        else:
            self.inter_gru = nn.GRU(intra_gru_hidden, inter_gru_hidden, bidirectional= False)        
            self.weight_W_inter = nn.Parameter(torch.Tensor(inter_gru_hidden ,inter_gru_hidden))
            self.bias_inter = nn.Parameter(torch.Tensor(inter_gru_hidden,1))
            self.weight_proj_inter = nn.Parameter(torch.Tensor(inter_gru_hidden, 1))
            self.final_linear = nn.Linear(inter_gru_hidden, n_classes)
        self.softmax_inter = nn.Softmax()
        self.final_softmax = nn.Softmax()
        self.weight_W_inter.data.uniform_(-0.1, 0.1)
        self.weight_proj_inter.data.uniform_(-0.1,0.1)
        self.bias_inter.data.uniform_(-0.1,0.1)
        
        
    def forward(self, intra_attention_vectors, state_inter):
        #pdb.set_trace()
        output_inter, state_inter = self.inter_gru(intra_attention_vectors, state_inter)        
        inter_squish = batch_matmul_bias(output_inter, self.weight_W_inter,self.bias_inter, nonlinearity='tanh')
        inter_attn = batch_matmul(inter_squish, self.weight_proj_inter)
        #inter_attn_norm = self.softmax_inter(inter_attn.transpose(1,0))
        inter_attn_norm = (inter_attn.transpose(1,0))
        inter_attn_vectors = attention_mul(output_inter, inter_attn_norm.transpose(1,0))        
        # final classifier
        final_map = self.final_linear(inter_attn_vectors.squeeze(0))
        return final_map, state_inter, inter_attn_norm
        #return F.log_softmax(final_map), state_inter, inter_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.inter_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.inter_gru_hidden))

class TextModels(nn.Module):

    def __init__(self, vocab_size, input_dim, model, num_layers, num_neurons, janossy_k,janossy_k2, device):
        """Create a model based on the request"""
        super(TextModels, self).__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.vocab_size = vocab_size
        self.original_input_dim = input_dim
        self.input_dim = int(input_dim/janossy_k)
        self.input_dim_mod = self.input_dim * janossy_k
        self.device = device
        self.model_name = model
        self.janossy_k = janossy_k
        self.janossy_k2 = janossy_k2
        # Define the loss function
        self.loss_func = nn.L1Loss()
    
        # Embedding Layer which is non trainable
        self.emb = nn.Embedding(self.vocab_size, self.original_input_dim)
        init.uniform_(self.emb.weight,a=-0.5,b=0.5)
        self.emb.weight.requires_grad = False
        # Create the model here based on input
        if self.model_name == 'lstm':
            self.model = nn.LSTM(10, 50,num_layers = 10, batch_first=True)#, bidirectional=True)
            self.model_activation = None
            self.model_out_shape = 50
        elif self.model_name == 'gru':
            gru_size = self.original_input_dim * janossy_k * janossy_k2
            self.model = nn.GRU( 500, 80, batch_first=True)
            #pdb.set_trace()
            self.model_activation = None
            self.model_out_shape = 80
        elif self.model_name == 'deepset':
            self.model = nn.Linear(self.original_input_dim, 30)#, bidirectional=True)
            self.model_activation = None
            self.model_out_shape = 30
        elif self.model_name == 'gru':
            gru_size = self.original_input_dim * janossy_k * janossy_k2
            self.model = nn.GRU( 500, 80, batch_first=True)
            #pdb.set_trace()
            self.model_activation = None
            self.model_out_shape = 80
        elif self.model_name == 'cnn':
            #self.model = nn.Conv2d(10,50, (3, 5, 10))
            self.layer1 = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(128, 20)
            self.model_out_shape = 20
            self.model_activation = None
            if torch.cuda.is_available():
                self.layer1.cuda()
                self.layer1.cuda()
                self.fc.cuda() 
        elif self.model_name == 'hats':
            small_hidden_size = 8
            small_n_layers = 2
            self.intra_attn = AttentionIntraRNN(batch_size=128, num_tokens=10, embed_size=10, intra_gru_hidden=20, bidirectional= True)
            self.intra_attn2 = AttentionIntraRNN(batch_size=128, num_tokens=10, embed_size=10, intra_gru_hidden=20, bidirectional= True)           
            self.inter_attn = AttentionInterRNN(batch_size=128, inter_gru_hidden=20, intra_gru_hidden=20, n_classes=20, bidirectional= True)        
            self.model_out_shape = 20   
            if torch.cuda.is_available():
                self.intra_attn.cuda()
                self.inter_attn.cuda()
            self.model_activation = None
        elif self.model_name == 'hier':
            small_hidden_size = 8
            small_n_layers = 2
            self.intra_attn = IntraRNN(batch_size=128, num_tokens=10, embed_size=10, intra_gru_hidden=20, bidirectional= True)         
            self.inter_attn = InterRNN(batch_size=128, inter_gru_hidden=20, intra_gru_hidden=20, n_classes=20, bidirectional= True)     
            self.model_out_shape = 20   
            if torch.cuda.is_available():
                self.intra_attn.cuda()
                self.inter_attn.cuda()
            self.model_activation = None
        else:
            self.model = nn.Linear(self.input_dim_mod, 30)
            self.model_activation = nn.Tanh()
            self.model_out_shape = 30
            init.xavier_uniform_(self.model.weight)
            self.model.bias.data.fill_(0)   
        
        # Multiple Hidden Layers based on input
        # Neurons in Hidden Layer based on input
        self.rho_mlp_linear = []
        for i in range(num_layers):
            if i == 0:
                self.rho_mlp_linear.append(nn.Linear(self.model_out_shape, num_neurons))
            else:
                self.rho_mlp_linear.append(nn.Linear(num_neurons, num_neurons))
            init.xavier_uniform_(self.rho_mlp_linear[-1].weight)
            self.rho_mlp_linear[-1].bias.data.fill_(0)
            self.rho_mlp_linear.append(nn.Tanh())
        if self.num_layers == 0:
            self.final_layer = nn.Linear(self.model_out_shape, 1)
        else:   
            for layer_num in range(len(self.rho_mlp_linear)):
                self.add_module("hidden_"+str(layer_num),self.rho_mlp_linear[layer_num])
            self.final_layer = nn.Linear(self.num_neurons, 1)
        init.xavier_uniform_(self.final_layer.weight)
        self.final_layer.bias.data.fill_(0)

    def func(tensor_a, tensor_b):
        return F.relu(tensor_a - tensor_b + 1) * F.relu(tensor_b - tensor_a + 1)
    def forward(self, input_tensor):
        """Lookup the tensor and then continue with feedforward"""
        # Input as a long tensor
        input_shape = input_tensor.shape
        emb_output = self.emb(input_tensor)
        emb_shape = emb_output.shape
        # Feed the obtained embedding to the Janossy Layer
        if self.model_activation is not None:
            #pdb.set_trace()
            model_out = self.model(emb_output)
            model_out = self.model_activation(model_out)
        else:   
            if self.model_name == 'hier':
                state_intra = self.intra_attn.init_hidden().cuda()
                state_inter = self.inter_attn.init_hidden().cuda()
                mini_batch = input_tensor
                mini_batch = mini_batch.view(mini_batch.shape[0],mini_batch.shape[1],-1)
                max_inters = mini_batch.shape[1]
                batch_size = mini_batch.shape[0]
                st = None
                sw = None
                for i in xrange(max_inters):
                    _st, state_intra = self.intra_attn(mini_batch[:,i,:].transpose(0,1), state_intra)
                    if(st is None):
                        st = _st
                    else:
                        st = torch.cat((st,_st),0)
                
                st = st.view(max_inters, batch_size, -1)
                model_out = self.inter_attn(st, state_inter)
            elif self.model_name == 'hats':
                state_intra = self.intra_attn.init_hidden().cuda()
                state_inter = self.inter_attn.init_hidden().cuda()
                mini_batch = input_tensor
                mini_batch = mini_batch.view(mini_batch.shape[0],mini_batch.shape[1],-1)
                max_inters = mini_batch.shape[1]
                batch_size = mini_batch.shape[0]
                st = None
                sw = None
                for i in xrange(max_inters):
                    _st, state_intra, _ = self.intra_attn(mini_batch[:,i,:].transpose(0,1), state_intra)
                    if(st is None):
                        st = _st
                    else:
                        st = torch.cat((st,_st),0)
                st = st.view(max_inters, batch_size, -1)
                model_out, state_inter, _ = self.inter_attn(st, state_inter)   
            elif self.model_name == 'cnn':
                out = self.layer1(emb_output)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                model_out = self.fc(out)
            elif self.model_name == 'deepset':
                emb_output = self.emb(input_tensor)
                emb_shape = emb_output.shape
                model_out = self.model(emb_output.view((-1, emb_output.shape[-1])))
                model_out = model_out.view(emb_shape[0], emb_shape[1], emb_shape[2],30 )
                model_out = torch.sum(model_out, 1)
                model_out = torch.sum(model_out, 1)
                #pdb.set_trace()
            else:
                emb_output = emb_output.view(emb_shape[0], -1, emb_shape[-1])
                model_out, _ = self.model(emb_output)
                model_out = model_out[:, -1, :]  # Just the final state
        if self.model_name in ['lstm','gru','cnn','hats','hier','hatsjanossy','hierjanossy','deepset']:
            rho_out = model_out
        else:
            summer_out = torch.sum(model_out, dim=1).to(self.device)
            rho_out = summer_out
        final_output = self.final_layer(rho_out)
        return final_output 

    def loss(self, input_tensor, output_tensor):
        """Loss Computations"""

        predicted_output = self.forward(input_tensor)
        return self.loss_func(predicted_output, output_tensor)

