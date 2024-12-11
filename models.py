import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Tokenizer
from transformers import T5EncoderModel

    
class SeperateT5EncoderClassifier(nn.Module):
    def __init__(self, size="base", num_labels=3, is_mental_health=False):
        super().__init__()
        if size == "base":
            in_features = 768
        elif size == "large":
            in_features = 1024
        
        heads = 1
        mid_features = in_features // 2
        self.is_mental_health = is_mental_health    

        model_dir = "/home/huima/data_ssd/huggingface_data/"
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir + "t5-" + size)
        self.context_encoder = T5EncoderModel.from_pretrained(model_dir + "t5-" + size)
        self.response_encoder = T5EncoderModel.from_pretrained(model_dir + "t5-" + size)

        self.dropout = nn.Dropout(0.1)
        self.attn = MultiHeadAttention(heads, in_features)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(mid_features, num_labels)
        )

    def _preprocess(self, context, response):
        "Preprocess data"
        if self.is_mental_health: #training empathy models for mental health dataset.
            max_len = 250
            data_context = [x for x in context] #" " is used for concatenete utters.
            data_response = [r for r in response]

            batch_context = self.tokenizer(data_context, max_length=max_len, padding=True, 
            truncation=True, return_tensors="pt")
            batch_response = self.tokenizer(data_response, max_length=max_len, padding=True, 
            truncation=True, return_tensors="pt")
        else:
            batch_context = self.tokenizer(context, max_length=200, padding=True, 
            truncation=True, return_tensors="pt")
            batch_response = self.tokenizer(response, max_length=50, padding=True, 
            truncation=True, return_tensors="pt")

        return batch_context, batch_response

    def forward(self, context, response):
        batch_context, batch_response = self._preprocess(context, response)
        context_attn_mask = batch_context["attention_mask"].cuda()
        response_attn_mask = batch_response["attention_mask"].cuda()

        context_outputs = self.context_encoder(input_ids=batch_context["input_ids"].cuda(), 
        attention_mask=batch_context["attention_mask"].cuda())["last_hidden_state"]
        
        response_outputs = self.response_encoder(input_ids=batch_response["input_ids"].cuda(), 
        attention_mask=batch_response["attention_mask"].cuda())["last_hidden_state"] #batch, len, dim

        # context-aware representation of the response post
        # query: response; key, value: context
        attn_mask = torch.matmul(response_attn_mask.unsqueeze(-1).float(), context_attn_mask.unsqueeze(1).float())
        response_outputs = response_outputs + self.dropout(self.attn(response_outputs, context_outputs, context_outputs, attn_mask))

        # mask average pooling
        response_mask_hidden_state = response_outputs * response_attn_mask.unsqueeze(-1)
        response_output = torch.sum(response_mask_hidden_state, dim=1)/torch.sum(response_attn_mask, dim=-1, keepdim=True)
        
        logits = self.classifier(response_output)
        return logits

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
		
        bs = q.size(0)
                
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
                
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output
	
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
            
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
