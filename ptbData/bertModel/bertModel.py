import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X,valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    if valid_lens.dim()==1:
        valid_lens=torch.repeat_interleave(valid_lens,X.shape[1])
    else:
        valid_lens=valid_lens.reshape(-1)
    print(valid_lens.shape)
    shape = X.shape
    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,  value=-1e6)

    return torch.softmax(X.reshape(shape),dim=-1)

class DotAttention(nn.Module):
    def __init__(self,drop_out):
        super().__init__()
        self.Dropout=nn.Dropout(drop_out)

    def forward(self,query,key,value,valid_lens):
        d=query.shape[-1]
        attention_weight=masked_softmax(torch.bmm(query,key.transpose(1,2))/torch.sqrt(torch.tensor(d, device=query.device)),valid_lens)
        return torch.bmm(self.Dropout(attention_weight),value)

class TransformerEncoderBlock(nn.Module):
    def __init__(self,num_hiddens,q_size,k_size,v_size,num_heads,ffn_input_size,ffn_num_hiddens,dropout=0):
        super().__init__()
        self.num_heads=num_heads
        self.W_q=nn.Linear(num_hiddens,q_size,bias=False)
        self.W_k=nn.Linear(num_hiddens,k_size,bias=False)
        self.W_v=nn.Linear(num_hiddens,v_size,bias=False)
        self.W_o=nn.Linear(num_hiddens,ffn_input_size)
        self.attention=DotAttention(dropout)
        self.norm=nn.LayerNorm(num_hiddens)   #指定末尾维度
        self.ffn=nn.Sequential(nn.Linear(ffn_input_size,ffn_num_hiddens),nn.ReLU(),nn.Linear(ffn_num_hiddens,num_hiddens))
    def forward(self,X,valid_len):
        '''
        X=[batch_size,max_len,hidden_size]
        valid_len=[batch_size]
        '''

        shape=X.shape
        Q=self.trans_qkv(self.W_q(X),self.num_heads)
        K=self.trans_qkv(self.W_k(X),self.num_heads)
        V=self.trans_qkv(self.W_v(X),self.num_heads)

        if valid_len is not None:
            valid_len=torch.repeat_interleave(valid_len,repeats=self.num_heads,dim=0)
        # X_a=self.attention(Q,K,V,valid_len).reshape(shape[0],self.num_heads,shape[1],-1).permute(0,2,1,3)
        X_a = self.attention(Q, K, V, valid_len) # 此时 shape 是 (B*H, seq_len, head_dim)
        X_a = X_a.permute(1, 0, 2).reshape(shape[1], shape[0], -1) # 合并多头
        X_a = X_a.permute(1, 0, 2) # 恢复成 (B, seq_len, hidden_size)
        X_a=X_a.reshape(shape[0],shape[1],-1)
        X_a=self.W_o(X_a)
        X_res=X+self.norm(X_a)
        return self.ffn(X_res)+X_res

    def trans_qkv(self,mat,num_heads):
        shape=mat.shape
        mat=mat.reshape(shape[0],shape[1],num_heads,-1)
        mat=mat.permute(0,2,1,3)
        mat=mat.reshape(-1,mat.shape[2],mat.shape[3])
        return mat

def set_tokens_format(tokens_a,tokens_b):
    tokens = ['<cls>'] + tokens_a + ['<seq>']
    segment=[0]*(len(tokens_a)+2)
    if tokens_b is not None:
        tokens+=tokens_b+['<seq>']
        segment+=[1]*(len(tokens_b)+1)
    return (tokens,segment)

class Bert(nn.Module):
    def __init__(self,vocab_size,hidden_size,ffn_input_size,num_layers,num_heads,ffn_num_hiddens,
                 max_len=1000,q_size=768,k_size=768,v_size=768,dropout=0):
        super().__init__()
        self.embeding=nn.Embedding(vocab_size,hidden_size)
        self.segment_embedding=nn.Embedding(2,hidden_size)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}",TransformerEncoderBlock(hidden_size,q_size,k_size,v_size,num_heads,ffn_input_size,ffn_num_hiddens,dropout))
        self.posEmbedding=nn.Parameter(torch.randn([1,max_len,hidden_size]))
        d2l.DotProductAttention

    def forward(self,X,segment,valid_len):
        X=self.embeding(X)+self.segment_embedding(segment)
        X=X+self.posEmbedding[:,:X.shape[1],:]
        for block in self.blks:
            X=block(X,valid_len)
        return X

class MaskLM(nn.Module):
    def __init__(self,num_hiddens,vocab_size,num_inputs=768):
        super().__init__()
        self.ffn=nn.Sequential(nn.Linear(num_inputs,num_hiddens),nn.ReLU(),nn.LayerNorm(num_hiddens),nn.Linear(num_hiddens,vocab_size))
    def forward(self,X,preposition):
        '''
        preposition=[batch_size,num_preposition]
        '''
        num_prepositon=preposition.shape[1]
        batch_size=X.shape[0]
        batch_idx=torch.arange(0,batch_size,device=X.device)
        batch_idx=torch.repeat_interleave(batch_idx,repeats=num_prepositon,dim=0)
        position_idx=preposition.reshape(-1)
        masked_X=X[batch_idx,position_idx]
        masked_X=masked_X.reshape([batch_size,num_prepositon,-1])
        return self.ffn(masked_X)

class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs):
        super().__init__()
        self.output = nn.Linear(num_inputs, 2)
    def forward(self, X):
    # X的形状：(batchsize,num_hiddens)
        return self.output(X)


class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_input,
        ffn_num_hiddens, num_heads, num_layers, dropout,
        max_len=1000, key_size=768, query_size=768, value_size=768,
        hid_in_features=768, mlm_in_features=768,
        nsp_in_features=768):
        super().__init__()
        self.encoder = Bert(vocab_size, num_hiddens,
            ffn_num_input,num_layers ,num_heads ,ffn_num_hiddens
            , max_len=max_len, k_size=key_size,
            q_size=query_size, v_size=value_size,dropout=dropout)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
            nn.Tanh())
        self.mlm = MaskLM(num_hiddens,vocab_size, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
    def forward(self, tokens, segments, valid_lens=None,
        pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat