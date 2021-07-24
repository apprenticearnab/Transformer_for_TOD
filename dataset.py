#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch 
import tokenizers
from tokenizers import ByteLevelBPETokenizer
from torch.nn.utils.rnn import pad_sequence

tokenizer = ByteLevelBPETokenizer('vocab.json','merges.txt')
tokenizer.add_special_tokens(["<PAD>","<|belief|>"," <|endofbelief|>","<|context|>"," <|user|> "," <|system|> ","<|endofcontext|>","<|action|>"," <|endofaction|>"])

pad_tok = tokenizer.encode("<PAD>").ids.pop()

def encoding(data_path_context,data_path_belief,maxlen):
    global torch,pad_sequence
    f=open(data_path_belief,'r',errors = 'ignore')
    raw=f.read()
    #raw=raw.lower()
    output = raw.split("\n")
    f1=open(data_path_context,'r',errors = 'ignore')
    raw=f1.read()
    #raw=raw.lower()
    input = raw.split("\n")
    srctgtpair = []
    src1 = []
    tgt1 = []
    length = []
    orig_out = [] 
    for j in range(len(input)):
          src1.append(torch.LongTensor(tokenizer.encode(input[j]).ids[:maxlen]))
          tgt1.append(torch.LongTensor(tokenizer.encode(output[j]).ids[:maxlen]))
    src = pad_sequence(src1,batch_first=True,padding_value=pad_tok)
    tgt = pad_sequence(tgt1,batch_first=True,padding_value=pad_tok)
    return src,tgt