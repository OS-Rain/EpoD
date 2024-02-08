import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class EpoD(BaseModel):
    def __init__(self, embed_dim, rnn_unit, num_layer, cheb_k, horizon, **args):
        super(EpoD, self).__init__(**args)
        self.node_embed = nn.Parameter(torch.randn(self.node_num, embed_dim), requires_grad=True)
        self.num_layer = num_layer
        self.horizon = horizon
        self.context_init_prompt = nn.Parameter(torch.empty(self.node_num, 64), requires_grad=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=256, dropout=0)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1)
        embed_dim = 10

        self.encoder_prompt = AVWDCRNN(self.input_dim-1, rnn_unit, cheb_k, embed_dim, 1, self.node_num)
        self.encoder_predict = AVWDCRNN(64+3, rnn_unit, cheb_k, embed_dim, 1, self.node_num)
        self.out_env = nn.Linear(2, 64)
        self.out_fc = nn.Linear(64, 1)
        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, rnn_unit), bias=True)
        self.bn = nn.BatchNorm2d(24)
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.kaiming_normal_(self.context_init_prompt)
        nn.init.kaiming_normal_(self.node_embed)
        nn.init.kaiming_normal_(self.end_conv.weight)
        nn.init.kaiming_normal_(self.out_env.weight)
        nn.init.kaiming_normal_(self.out_fc.weight)

    def forward(self, source, label=None):  # (b, t, n, f)
        # ================================================================================================
        bs, t, node_num, _ = source.shape
        init_state = self.encoder_prompt.init_hidden(bs, node_num) # torch.Size([2, 32, 307, 64])
        output, _ = self.encoder_prompt(source[:, :, :, 1:], init_state, self.node_embed, True, None)
        prompt_nodes = self.transformer_decoder(self.context_init_prompt.unsqueeze(0).expand(bs * t, self.node_num, -1), output.view(-1, node_num, 64)).view(bs , t, node_num, 64)
        source_high = self.out_env(source[:, :, :, 1:])
        kl_loss = torch.kl_div(prompt_nodes.softmax(dim=-1).log(),source_high.softmax(dim=-1)).mean()
        source = torch.cat([source, prompt_nodes], dim=3)
        output, _ = self.encoder_predict(source, init_state, self.node_embed, False, prompt_nodes) #[b, t, n, 64]
        pred_x = self.out_fc(F.relu(prompt_nodes))
        output = output[:, -1:, :, :]
        pred = self.end_conv(output)
        # ===========================================================================================
        # print("The shape of pred is ", pred.shape)
        # print("The shape of pred_x is ", pred_x.shape)
        return pred, pred_x, kl_loss

 
class AVWDCRNN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_layer, num_nodes):
        super(AVWDCRNN, self).__init__()
        assert num_layer >= 1, 'At least one DCRNN layer in the Encoder.'
        self.input_dim = dim_in
        self.num_layer = num_layer
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layer):
            self.dcrnn_cells.append(AGCRNCell(dim_out, dim_out, cheb_k, embed_dim))


    def forward(self, x, init_state, node_embed, is_propmt, prompt_answer):
        seq_length = x.shape[1]
        current_inputs = x
        bs = x.shape[0]
        node_num = x.shape[2]
        output_hidden = []
        for i in range(self.num_layer):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = state.to(x.device)
                if(is_propmt):
                    state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embed, None)
                else:
                    state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state + prompt_answer[:, t, :, :], node_embed, prompt_answer[:, t, :, :])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden


    def init_hidden(self, batch_size, node_num):
        init_states = []
        for i in range(self.num_layer):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, node_num))
        return torch.stack(init_states, dim=0)


class AGCRNCell(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)
        

    def forward(self, x, state, node_embed, prompt_answer):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) # 64 + 3/64 = 67/128
        z_r = self.gate(input_and_state, node_embed, prompt_answer)
        # z_r = self.bn1(z_r)
        z_r = torch.sigmoid(z_r) # (67/128, 10)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        
        candidate = torch.cat((x, z*state), dim=-1)
        hc = self.update(candidate, node_embed, prompt_answer)
        # hc = self.bn2(hc)
        hc = torch.tanh(hc)
        h = r*state + (1-r)*hc
        # h = self.dropout(h)
        return h

    def init_hidden_state(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, 2, dim_in, dim_out))# [10, 2, 128, 64] / [10, 2, 67, 64]
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.reset_parameter()    
    
    def reset_parameter(self):
        nn.init.kaiming_normal_(self.weights_pool)
        nn.init.kaiming_normal_(self.bias_pool)

    def forward(self, x, node_embed, prompt_answer): #  x [32, 170, 128] / [32, 170, 67]; node_embed [170, 10]
        node_num = node_embed.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embed, node_embed.transpose(0, 1))), dim=1) # [170, 170] 
        support_set = [torch.eye(node_num).to(supports.device), supports]
        supports = torch.stack(support_set, dim=0) # [2, 170, 170]
        weights = torch.einsum('nd,dkio->nkio', node_embed, self.weights_pool) # [170, 2, 128, 64] / [170, 2, 67, 64]
        bias = torch.matmul(node_embed, self.bias_pool)

        if (prompt_answer != None):
            sub_support = F.softmax(-F.relu(torch.cdist(prompt_answer, prompt_answer, p=1)), dim=1)
            sub_support = sub_support.unsqueeze(dim=1).expand(-1, 2, -1, -1)            
            supports = torch.einsum("knm,bkmc->bknc", supports, sub_support)
            x_g = torch.einsum("bknm,bmc->bknc", supports, x)
        else:
            x_g = torch.einsum("knm,bmc->bknc", supports, x) # [32, 2, 170, 128] / [32, 2, 170, 67]
        
        x_g = x_g.permute(0, 2, 1, 3) # [32, 170, 2, 67]
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias # [32, 170, 64]
        return x_gconv