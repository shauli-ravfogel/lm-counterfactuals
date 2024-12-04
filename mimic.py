import torch
import torch.nn as nn
import sk2torch

class InterventionModule(nn.Module):
    def __init__(self, mean_0, mean_1, A, mlp, alpha=1.0):
        super().__init__()
        self.mean_0 = torch.tensor(mean_0.astype("float32"))
        self.mean_1 = torch.tensor(mean_1.astype("float32"))
        self.A = torch.tensor(A.astype("float32"))
        #self.mlp = sk2torch.wrap(mlp)
        self.alpha = alpha
        # set requires_grad=False to all params of the mlp
        #for p in self.mlp.parameters():
        #    p.requires_grad = False
    
    def to_cuda(self, device):
      self.A = self.A.to(device)
      self.mean_0 = self.mean_0.to(device)
      self.mean_1 = self.mean_1.to(device)
      #self.mlp = self.mlp.to(device)

    def to_cpu(self):
      self.A = self.A.to("cpu")
      self.mean_0 = self.mean_0.to("cpu")
      self.mean_1 = self.mean_1.to("cpu")
      #self.mlp = self.mlp.to("cpu")

    def forward(self, hidden_states):
        self.to_cuda(hidden_states.device)
        # if hidden state is half, convert laso the params to half precision
        if hidden_states.dtype == torch.float16:
          self.A = self.A.half()
          self.mean_0 = self.mean_0.half()
          self.mean_1 = self.mean_1.half()

        x = hidden_states.clone().reshape(-1, hidden_states.shape[-1])
        if self.alpha != 0:
            x = self.alpha*self.mean_1 + (x - self.alpha*self.mean_0)@self.A
            #x = self.A * x + self.mean_1 - self.A * self.mean_0
        x = x.reshape(hidden_states.shape)
        return x


def insert_intervention(model, model_name, intervention, layer, after_layer_norm=False, replace_existing=False):
    if "gpt2" in model_name.lower():
        # if the mlp is already a Sequential object, do nothing
        if isinstance(model.transformer.h[layer].mlp, torch.nn.Sequential):
            return
        if not replace_existing:
            model.transformer.h[layer].mlp = torch.nn.Sequential(model.transformer.h[layer].mlp, intervention)
        else:
            
            model.transformer.h[layer].mlp = torch.nn.Sequential(model.transformer.h[layer].mlp[:-1], intervention)

    elif "llama" in model_name.lower():
        # if the mlp is already a Sequential object, do nothing
        if isinstance(model.model.layers[layer].post_attention_layernorm, torch.nn.Sequential):
            return
        if not replace_existing:
            model.model.layers[layer].post_attention_layernorm = torch.nn.Sequential(model.model.layers[layer].post_attention_layernorm, intervention)
        else:
            model.model.layers[layer].post_attention_layernorm = torch.nn.Sequential(model.model.layers[layer].post_attention_layernorm[0], intervention)

    else:
        raise NotImplementedError("Only GPT2 is supported")


def remove_intervention(model, model_name, layer):
    if "gpt2" in model_name.lower():
        # if the mlp is not a Sequential object, do nothing
        if not isinstance(model.transformer.h[layer].mlp, torch.nn.Sequential):
            return
        model.transformer.h[layer].mlp = model.transformer.h[layer].mlp[0]

    elif "llama" in model_name.lower():
        # if the mlp is not a Sequential object, do nothing
        if not isinstance(model.model.layers[layer].post_attention_layernorm, torch.nn.Sequential):
            return
        model.model.layers[layer].post_attention_layernorm = model.model.layers[layer].post_attention_layernorm[0]

    else:
        raise NotImplementedError("Only GPT2 is supported")
