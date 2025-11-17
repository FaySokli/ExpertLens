import torch
from torch import clamp as t_clamp
from torch import nn as nn
from torch import sum as t_sum
from torch import max as t_max
from torch import einsum
from torch.nn import functional as F

class Specializer(nn.Module):
    def __init__(self, hidden_size, device):
        super(Specializer, self).__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        self.embedding_changer_1 = nn.Linear(self.hidden_size, self.hidden_size//2).to(self.device)
        self.embedding_changer_4 = nn.Linear(self.hidden_size//2, self.hidden_size).to(self.device)
        
    def forward(self, embs):
        embs_1 = F.relu(self.embedding_changer_1(embs)).to(self.device)
        embs_2 = self.embedding_changer_4(embs_1).to(self.device)
        
        return embs_2

class DeepViewClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, device):
        super(DeepViewClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device
        self.cls_4 = nn.Linear(self.hidden_size, self.hidden_size//2).to(device)
        self.cls_5 = nn.Linear(self.hidden_size//2, self.num_classes).to(device)
        
    def forward(self, embedding):
        x = F.relu(self.cls_4(embedding))
        out = self.cls_5(x)
        return out

class MoEBiEncoder(nn.Module):
    def __init__(
        self,
        doc_model,
        tokenizer,
        num_classes,
        max_tokens,
        normalize,
        specialized_mode,
        pooling_mode,
        use_adapters,
        device,
    ):
        super(MoEBiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.hidden_size = self.doc_model.config.hidden_size
        self.tokenizer = tokenizer
        self.device = device
        self.normalize = normalize
        self.max_tokens = max_tokens
        self.use_adapters = use_adapters
        assert specialized_mode in ['densec3_top1', 'densec3_w'], 'Only densec3_top1 and densec3_w specialzed modes allowed'
        self.specialized_mode = specialized_mode
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        if pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif pooling_mode == 'identity':
            self.pooling = self.identity
        
        self.num_classes = num_classes
        
        self.specializer = nn.ModuleList([Specializer(self.hidden_size, self.device) for _ in range(self.num_classes)])    


    def encoder_no_moe(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
        
    
    def encoder(self, sentences, logits):
    # def encoder(self, sentences):
        embedding = self.encoder_no_moe(sentences)
        if self.use_adapters:
            # logits = self.cls(embedding).to(self.device)
            # cls_logits = self.cls(logits).to(self.device)
            embedding = self.embedder(embedding, logits)
        return embedding
    
    def cls(self, out):
        if self.training:
            out = torch.softmax(out/10, dim=-1)
            # out = torch.sigmoid(out/10)

            # TOP-k GATING
            topk_values, topk_indices = torch.topk(out, 1, dim=1)
            mask = torch.zeros_like(out).scatter_(1, topk_indices, 1)
            out = out * mask
            return out
        
        else:
            if self.specialized_mode == 'densec3_top1':
                out = torch.softmax(out/10, dim=-1)
                # out = torch.sigmoid(out/10)

                # TOP-k GATING
                topk_values, topk_indices = torch.topk(out, 1, dim=1)
                mask = torch.zeros_like(out).scatter_(1, topk_indices, 1)
                
                # Multiply the original output with the mask to keep only the max value
                out = out * mask
                return out
            
            elif self.specialized_mode == 'densec3_w':
                out = torch.softmax(out/10, dim=-1)
                # out = torch.sigmoid(out/10)
                return out
    

    def forward(self, data):
        logits_class = self.cls(data[2]).to(self.device)
        pos_embedding = self.encoder(data[1], logits_class).to(self.device)

        if self.specialized_mode == 'densec3_top1':
            query_embedding = self.encoder(data[0], logits_class).to(self.device)
        elif self.specialized_mode == 'densec3_w':
            query_embedding = self.encoder_no_moe(data[0]).to(self.device)
            if self.use_adapters:
                query_embedding = self.embedder_q(query_embedding).to(self.device)

        return query_embedding, pos_embedding

    def val_forward(self, data):
        logits_class = self.cls(data[2]).to(self.device)
        pos_embedding = self.encoder(data[1], logits_class).to(self.device)

        if self.specialized_mode == 'densec3_top1':
            query_embedding = self.encoder(data[0], logits_class).to(self.device)
        elif self.specialized_mode == 'densec3_w':
            query_embedding = self.encoder_no_moe(data[0]).to(self.device)
            if self.use_adapters:
                query_embedding = self.embedder_q(query_embedding).to(self.device)

        return query_embedding, pos_embedding


    def embedder(self, embedding, logits_class):
        embs = [self.specializer[i](embedding) for i in range(self.num_classes)]

        embs = torch.stack(embs, dim=1).to(self.device)
        
        embs = (F.normalize(einsum('bmd,bm->bd', embs, logits_class), dim=-1, eps=1e-6) + embedding).to(self.device)
        
        if self.normalize:
            return F.normalize(embs, dim=-1).to(self.device)
        return embs
    
    def embedder_q(self, embedding):
        embs = [self.specializer[i](embedding) for i in range(self.num_classes)]
        embs = torch.stack(embs, dim=1)
        
        aggregated_embs = torch.mean(embs, dim=1)
        aggregated_embs = (F.normalize(aggregated_embs, dim=-1) + embedding).to(self.device)

        if self.normalize:
            aggregated_embs = F.normalize(aggregated_embs, dim=-1)
        return aggregated_embs
    
    def embedder_q_inf(self, embedding):
        embs = [self.specializer[i](embedding) for i in range(self.num_classes)]
        embs = torch.stack(embs, dim=1)
        
        return F.normalize(embs, dim=-1)
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output["last_hidden_state"]
        # last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden[:, 0]


    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
