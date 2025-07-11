# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 19:40
@author: LiFan Chen
@Filename: model_glu.py
@Software: PyCharm
"""
# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/7 13:40
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
#import main_glu


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.device = device

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        # if torch.cuda.is_available():
        #     self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        # else:
        #     self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
        # 在正确的 device 上初始化 scale
        self.scale = torch.sqrt(
                torch.FloatTensor([hid_dim // n_heads])
                ).to(self.device)

    def forward(self, query, key, value, mask=None):
        # if len(query.shape) > len(key.shape):
        #     bsz = query.shape[0]
        # else:
        #     bsz = key.shape[0]
        bsz = query.shape[0] if query.dim() > key.dim() else key.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # Q, K = Q.cpu(), K.cpu()
        # del Q, K
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        return self.fc(
            torch.matmul(self.do(F.softmax(energy, dim=-1)), V).permute(0, 2, 1, 3).contiguous().view(bsz, -1,
                                                                                                      self.n_heads * (
                                                                                                              self.hid_dim // self.n_heads)))


class Encoder(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)

    def forward(self, protein):
        # pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        # protein = protein + self.pos_embedding(pos)
        # protein = [batch size, protein len,protein_dim]

        conv_input = self.fc(protein)

        # conv_input=[batch size,protein len,hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, protein len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, protein len]

            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, protein len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg1 = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg1 = self.ln(trg1 + self.do(self.ea(trg1, src, src, src_mask)))
        trg1 = self.ln(trg1 + self.do(self.pf(trg1)))
        src1 = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        src1 = self.ln(src1 + self.do(self.ea(src1, trg, trg, trg_mask)))
        src1 = self.ln(src1 + self.do(self.pf(src1)))
        trg, src = trg.cpu(), src.cpu()
        del trg, src, trg_mask, src_mask

        trg1attn = trg1

        m1 = torch.mean(trg1, 1)
        trg1 = torch.unsqueeze(m1, 1)
        m2 = torch.mean(src1, 1)
        src1 = torch.unsqueeze(m2, 1)

        return trg1, src1, trg1attn


# IMT

class Decoder(nn.Module):
    """ compound feature extraction."""

    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        for idx, layer in enumerate(self.layers):
            trg, src, trgattn = layer(trg, src, trg_mask, src_mask)
            if idx == 0:
                trgattn_fina = trgattn
        del trg_mask, src_mask
        return trg, src, trgattn_fina


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features  
        self.dropout = dropout  
        self.alpha = alpha  
        self.concat = concat  

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)

        a_input = self._prepare_attentional_mechanism_input(Wh)  
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  

        zero_vec = -9e15 * torch.ones_like(e)  
        attention = torch.where(adj > 0, e, zero_vec)  
        attention = F.softmax(attention, dim=1) 
        attention = F.dropout(attention, self.dropout, training=self.training)  
        h_prime = torch.matmul(attention, Wh)  

        if self.concat:
            return F.elu(h_prime)  
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  
        Wh_repeated_alternating = Wh.repeat(N, 1)  
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)  
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):
        def __init__(self, atom_dim, hid_dim, gat_heads, dropout, alpha, n_layers, device):
        super(GAT, self).__init__()

        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim) for _ in range(n_layers)])
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(n_layers)])

        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.atom_dim = atom_dim
        self.attentions = [GraphAttentionLayer(atom_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(gat_heads)]  
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hid_dim * gat_heads, atom_dim, dropout=dropout, alpha=alpha,
                                           concat=False)  

    def forward(self, x, adj, n_layers):
        # x = F.dropout(x, self.dropout, training=self.training)
        # # x = self.embedding_layer_atom(x)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)

        for i in range(n_layers):
            h = torch.relu(self.W_gnn[i](x))
            size = h.size()[0]  # batch:1
            N = h.size()[1]  

            h1 = h.repeat(1, 1, N)  # h:(1,78,34)   h1:(1,78,34*78), 34*78=2652
            h2 = h1.view(size, N * N, -1)  # h2:(1,78*78,34)  78*78=6084    (78*34*78)/(78*78)=34
            h3 = h.repeat(1, N, 1)  # h3:(1,78*78,34) , 78*78=6084
            h4 = torch.cat([h2, h3], dim=2)  # h4:(1,78*78,2*34)
            a_input = h4.view(size, N, -1, 2 * self.atom_dim)  # a_input:(1,78,78,68)

            # a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1, 2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec) 
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.matmul(attention, h)
            x = x + h_prime  # (1,78,34)
        # return torch.unsqueeze(torch.mean(x, 1), 1)    # (1,78,34)——>(1,34)——>(1,1,34)
        return x


# BERT

class Embedding(nn.Module):
    def __init__(self, device):
        super(Embedding, self).__init__()
        self.device = device
        self.tok_embed = nn.Embedding(vocab_size,
                                      d_model)  # token embedding (look-up table)    # vocab_size=8393  d_model=32
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding    # max_len = 2048  (8112)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=self.device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        embedding = self.pos_embed(pos)  
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k))
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]

        import pandas as pd
        output_attn = attn.detach().squeeze().cpu().numpy()
        np.savez('output_attn.npz', a=output_attn)

        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class BERT(nn.Module):
    def __init__(self, n_word, device):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, hid_dim, vocab_size
        max_len = 8112  # human 8112  C.elegan 8377
        n_layers = 3
        n_head = 8
        d_model = 32  
        d_ff = 64
        d_k = 32
        d_v = 32
        hid_dim = 64
        vocab_size = n_word

        self.embedding = Embedding(device)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, hid_dim)
        # self.fc_task = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 2),
        # )
        # self.classifier = nn.Linear(2, 2)

    def forward(self, input_ids):
        # input_ids[batch_size, seq_len] like[8,1975]
        output = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)

        import pandas as pd
        output1 = pd.DataFrame(output.detach().squeeze().cpu().numpy())
        output1.to_excel('output1.xlsx', index=False)

        output = self.fc(output)
        # m = torch.mean(output,1)
        # output = torch.unsqueeze(m,1)
        return output


class InteractionModel(nn.Module):
    def __init__(self, hid_dim, n_heads):
        super(InteractionModel, self).__init__()
        # self.compound_embedding = nn.Linear(compound_feature_size, hidden_size)
        # self.protein_embedding = nn.Linear(protein_feature_size, hidden_size)
        self.compound_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.protein_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.compound_fc = nn.Linear(hid_dim, hid_dim)
        self.protein_fc = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.ReLU()

        self.hid_dim = hid_dim
        # self.setup_weights()
        # self.init_parameters()



    def forward(self, compound_features, protein_features):
        compound_embedded = self.activation(compound_features)
        protein_embedded = self.activation(protein_features)

        compound_embedded = compound_embedded.permute(1, 0, 2)
        protein_embedded = protein_embedded.permute(1, 0, 2)

        compound_attention_output, _ = self.compound_attention(compound_embedded, compound_embedded,
                                                               compound_embedded)
        protein_attention_output, _ = self.protein_attention(protein_embedded, protein_embedded, protein_embedded)

        compound_attention_output = compound_attention_output.permute(1, 0, 2)
        protein_attention_output = protein_attention_output.permute(1, 0, 2)

        compound_output = self.activation(self.compound_fc(compound_attention_output))
        protein_output = self.activation(self.protein_fc(protein_attention_output))

        com_att = torch.unsqueeze(torch.mean(compound_output, 1), 1)
        pro_att = torch.unsqueeze(torch.mean(protein_output, 1), 1)
        return com_att, pro_att




# NTN  (tensor_network)

class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, k_feature, hid_dim, k_dim):
        super(TensorNetworkModule, self).__init__()
        self.k_feature = k_feature
        self.hid_dim = hid_dim
        self.k_dim = k_dim

        self.setup_weights()
        self.init_parameters()

        self.fc1 = nn.Linear(hid_dim, k_dim)
        self.fc2 = nn.Linear(k_dim, hid_dim)

    def setup_weights(self):
        """
        Defining weights.  k_feature = args.filters_3   args.tensor_neurons = k_dim
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.k_feature, self.k_feature, self.k_dim
            )
        )  # (16,16,16)
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.k_dim, 2 * self.k_feature)
        )  # (16,32)
        self.bias = torch.nn.Parameter(torch.Tensor(self.k_dim, 1))  # (16,1)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.   com_att
        :param embedding_2: Result of the 2nd embedding after attention.   pro_att
        :return scores: A similarity score vector.
        """
        embedding_1 = torch.squeeze(embedding_1, dim=1)  # (1,1,64)——>(1,64)   (batch_size,1,64)——>(batch_size,64)
        embedding_1 = self.fc1(embedding_1)  # (1,64)——>(1,16)
        embedding_2 = torch.squeeze(embedding_2, dim=1)
        embedding_2 = self.fc1(embedding_2)

        batch_size = len(embedding_1)
        # print(self.weight_matrix.view(self.k_feature, -1).shape) 
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.k_feature, -1)
        )
        # print(scoring.shape)
        scoring = scoring.view(batch_size, self.k_feature, -1).permute([0, 2, 1])  
        # print(scoring.shape)
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.k_feature, 1)
        ).view(batch_size, -1)
        # print(scoring.shape)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        # print(combined_representation.shape)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )  # torch.t:转置
        # print(block_scoring.shape)
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        # print(scores.shape)    # (1,16)——(batch_size,16)
        scores = torch.unsqueeze(scores, 1)  # (1,16) ——> (1,1,16)
        scores = self.fc2(scores)  # (1,1,16) ——> (1,1,64)
        return scores


class Predictor(nn.Module):
    def __init__(self, gat, bert, decoder, inter_att, tensor_network, device, n_fingerprint, n_layers, atom_dim=34):
        super().__init__()

        self.embed_fingerprint = nn.Embedding(n_fingerprint, atom_dim)
        self.gat = gat
        self.Bert = bert
        self.inter_att = inter_att
        self.tensor_network = tensor_network
        # self.embed_word = nn.Embedding(n_word, atom_dim)

        # self.encoder = encoder
        self.n_layers = n_layers
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(34, 34))
        self.init_weight()

        self.protein_dim = 100
        self.hid_dim = 64
        self.atom_dim = 34
        self.fc1 = nn.Linear(self.protein_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.atom_dim, self.hid_dim)

        self.W_attention = nn.Linear(self.hid_dim, self.hid_dim)


        self.out = nn.Sequential(
            nn.Linear(self.hid_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, compound1, compound2, adj, protein1, protein2, n_layers):
        # compound = [atom_num, atom_dim]
        # adj = [atom_num, atom_num]
        # protein = [protein len, 100]
        # 原compound = self.gcn(compound, adj)
        # global num_node
        # num_node = compound.shape[0]
        # compound_vectors = self.embed_compound(compound)
        # compound_vectors = nn.Embedding(num_node, atom_dim)

        protein1 = torch.unsqueeze(protein1,
                                   dim=0)  # (1,804,100)    # protein1 =[ batch size=1,protein len, protein_dim]
        protein1 = self.fc1(protein1)  # (1,804,64)     # protein1 =[ batch size=1,protein len, hid_dim]

        compound1 = torch.unsqueeze(compound1, dim=0)  # (1,1,34)       # compound1 = [batch size=1 ,atom_num, atom_dim]
        compound1 = self.fc2(compound1)  # (1,1,64)       # compound1 = [batch size=1 ,atom_num, hid_dim]

        protein1_c, compound1_p , trgattn = self.decoder(protein1,
                                               compound1)  # protein1_c:(1,1,64)


        compound_vectors = self.embed_fingerprint(compound2)  # compound_vectors:(78,34)
        compound2 = torch.unsqueeze(compound_vectors, dim=0)  # 1,78,34
        compound2 = self.gat(compound2, torch.unsqueeze(adj, 0), n_layers)  # 1,78,34
        compound2 = self.fc2(compound2)  # compound_vectors:(1,1,64)

        # enc_src = self.encoder(protein1)
        # enc_src = [batch size, protein len, hid dim

        protein2 = self.Bert(protein2.unsqueeze(0))  # protein2:(1,806,64)  # [batch size, protein len, hid dim]
        # protein2_att = self.attention_cnn(compound2, protein2, n_layers)
        com_att, pro_att = self.inter_att(compound2, protein2)
        scores = self.tensor_network(com_att, pro_att)
        # out_fc = torch.cat((compound2, protein2_att, compound1_p, protein1_c), 2)
        out_fc = torch.cat((scores, compound1_p, protein1_c), 2)
        out = self.out(out_fc)  # out = [batch size, 2]
        out = torch.squeeze(out, dim=0)
        return out , trgattn

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]

        compound1, compound2, adj, protein1, protein2 = inputs
        Loss = nn.CrossEntropyLoss()

        if train:

            predicted_interaction = self.forward(compound1, compound2, adj, protein1, protein2, n_layers)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            predicted_interaction , trgattn = self.forward(compound1, compound2, adj, protein1, protein2, n_layers)
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0, 1]
            return correct_labels, predicted_labels, predicted_scores ,trgattn


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        for data in dataset:
            i = i + 1
            loss = self.model(data)
            loss = loss / self.batch
            loss.backward()
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for idx, data in enumerate(dataset):
                correct_labels, predicted_labels, predicted_scores ,trgattn= self.model(data, train=False)
                if idx ==0 :
                    trgattn = trgattn.squeeze().cpu()
                    print(trgattn.shape)
                    import pandas as pd
                    df = pd.DataFrame(trgattn)
                    df.to_csv("./trgattn.csv", index=False)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
        # AUC = roc_auc_score(T, S)
        # precision = precision_score(T, Y)
        # recall = recall_score(T, Y)
        # return AUC, precision, recall
        return S

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)