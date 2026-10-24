import torch
from d2l import torch as d2l
import math
import os
import numpy
import random
from ptbData import *
import pandas as pd
Data_dir='./data/train-00000-of-00001.parquet'

def _read_wiki(data_dir):
    df = pd.read_parquet(data_dir)
    lines = df['text'].tolist()
    lines=[line.strip().lower().split('.') for line in lines if len(line.split(' . '))>1]
    random.shuffle(lines)
    return lines


def get_next_sentence(sentence,nxt_sentence,lines):
    if random.random()<0.5:
        is_next=True
    else:
        nxt_sentence=random.choice(random.choice(lines))
        is_next=False
    return (sentence,nxt_sentence,is_next)

def get_nsp_data(line,lines,max_len):
    nsp_line=[]
    for i in range(len(line)-1):
        token_a,token_b,is_next=get_next_sentence(line[i],line[i+1],lines)
        if(len(token_a)+len(token_b)+3>max_len):
            continue
        tokens,segments=d2l.get_tokens_and_segments(token_a,token_b)
        nsp_line.append((tokens,segments,is_next))
    return nsp_line
def _replace_masked_tokens(tokens,candidate_replace_position,num_prepositon,vocab):
    masked_tokens=[token for token in tokens]  #list的深拷贝
    random.shuffle(candidate_replace_position)
    pre_positon_and_truth=[]
    for i,idx_prepositon in enumerate(candidate_replace_position):
        if i>=num_prepositon:
            break
        mask_symbol=None
        if random.random()<0.8:
            mask_symbol='<mask>'
        else:
            if random.random()<0.5:
                mask_symbol=tokens[idx_prepositon]
            else:
                mask_symbol=random.choice(vocab.idx_to_str)
        masked_tokens[idx_prepositon]=mask_symbol
        pre_positon_and_truth.append((idx_prepositon,tokens[idx_prepositon]))
    return masked_tokens,pre_positon_and_truth

def _get_mlm_tokens(tokens,vocab):
    candidate_preposition=[]
    for i,token in enumerate(tokens):
        if token in ('<cls>','<sep>'):
            continue
        candidate_preposition.append(i)
    num_prepositon=max(1,round(len(candidate_preposition)*0.15))
    masked_tokens,preposition_and_truth=_replace_masked_tokens(tokens,candidate_preposition,num_prepositon,vocab)
    preposition_and_truth=sorted(preposition_and_truth,key=lambda x:x[0])
    idx_prepositons=[v[0] for v in preposition_and_truth]
    truth_preposition=[vocab[v[1]] for v in preposition_and_truth]
    return vocab[masked_tokens],idx_prepositons,truth_preposition

def _pad_bert_tokens(examples,max_len,vocab):
    all_tokens_id,all_valid_lens,all_segments=[],[],[]
    all_mlm_valid_lens,all_mlm_labels,all_mlm_preposition=[],[],[]
    all_nsp_labels=[]
    mlm_max_len=round(max_len*0.15)
    for token_ids,idx_preposition,label_preposition,segment,is_next in examples:
        all_tokens_id.append(torch.tensor(token_ids+[vocab['<pad>']]*(max_len-len(token_ids)),dtype=torch.long))
        all_valid_lens.append(torch.tensor(len(token_ids),dtype=torch.long))
        all_segments.append(torch.tensor(segment+[0]*(max_len-len(segment)),dtype=torch.long))
        all_mlm_labels.append(torch.tensor(label_preposition+[0]*(mlm_max_len-len(label_preposition)),dtype=torch.long))
        all_mlm_valid_lens.append(torch.tensor([1]*len(label_preposition)+[0]*(mlm_max_len-len(label_preposition)),dtype=torch.long))
        all_mlm_preposition.append(torch.tensor(idx_preposition+[0]*(mlm_max_len-len(idx_preposition)),dtype=torch.long))
        all_nsp_labels.append(torch.tensor(is_next,dtype=torch.long))
    return (all_tokens_id,all_segments,all_nsp_labels,all_valid_lens,
            all_mlm_preposition,all_mlm_labels,all_mlm_valid_lens)

class _wikiTextDataSet(torch.utils.data.Dataset):
    def __init__(self,paragraphs,max_len):
        paragraphs=[d2l.tokenize(paragraph,token='word') for paragraph in paragraphs]
        sentences=[line for paragraph in paragraphs for line in paragraph]
        self.vocab=Vocab(sentences,10,reserved_tokens=['<mask>','<cls>','<sep>','<pad>'],tokenize=True)
        examples=[]
        for paragraph in paragraphs:
            examples.extend(get_nsp_data(paragraph,paragraphs,max_len))
        examples=[(_get_mlm_tokens(tokens,self.vocab))+(segments,is_next) for tokens,segments,is_next in examples]
        (self.all_tokens_id,self.all_segments,self.all_nsp_labels,self.all_valid_lens,
            self.all_mlm_preposition,self.all_mlm_labels,self.all_mlm_valid_lens)=_pad_bert_tokens(examples,max_len,self.vocab)


    def __getitem__(self,idx):
        return (self.all_tokens_id[idx],self.all_segments[idx],self.all_valid_lens[idx], self.all_mlm_preposition[idx],
                self.all_mlm_valid_lens[idx],self.all_mlm_labels[idx], self.all_nsp_labels[idx])

    def __len__(self):
        return len(self.all_tokens_id)


def load_data_wiki(batch_size,max_len):
    global Data_dir
    num_workers=d2l.get_dataloader_workers()
    paragraphs=_read_wiki(Data_dir)
    train_set=_wikiTextDataSet(paragraphs,max_len)
    train_iter=torch.utils.data.DataLoader(train_set,batch_size,shuffle=True,num_workers=num_workers)
    return train_iter,train_set.vocab
        
