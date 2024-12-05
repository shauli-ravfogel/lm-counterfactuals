import numpy as np
import tqdm
import scipy
from scipy.stats import gumbel_l, gumbel_r
from transformers.generation import LogitsProcessor,LogitsProcessorList
import torch
from scipy.special import softmax, logsumexp

class GumbelProcessor(LogitsProcessor):
    def __init__(self, precomputed_noise=None,noise=1, replaced_pairs=None):
        self.precomputed_noise = precomputed_noise
        self.i=0
        self.replaced_pairs=replaced_pairs
        #np.random.seed(noise)
        self.noises = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.precomputed_noise is not None:
            if self.i < len(self.precomputed_noise):
                out = scores + self.precomputed_noise[self.i]
            else:
                gumbel = torch.tensor(np.random.gumbel(loc=0.0, scale=1.0, size=scores.shape)).to(scores.device)
                out = scores + gumbel
            self.i += 1
            return out
        

def truncated_gumbel_vectorized(logalpha, truncation):
    """
    Vectorized version of truncated Gumbel sampling.
    """
    gumbel = np.random.gumbel(size=len(logalpha)) + logalpha #np.log(alpha)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))

def topdown_vectorized(alphas, k, logsumalphas, logalphas):
    """
    Vectorized version of topdown function.
    """
    topgumbel = np.random.gumbel() + logsumalphas
    truncated_gumbels = truncated_gumbel_vectorized(logalphas, topgumbel)
    gumbels = truncated_gumbels
    gumbels[k] = topgumbel
    gumbels -= logalphas #np.log(alphas) 
    return gumbels


def counterfactual_generation_vectorized(model, tokenizer, prompt, sentence):
    vocab_size = len(tokenizer.get_vocab())
    tokens_sentence = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=False).to(model.device)
    tokens_prompt = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    tokens = torch.cat((tokens_prompt, tokens_sentence), dim=1).long()
    len_prompt = len(tokens_prompt[0])

    with torch.no_grad():
        logits_all = model(tokens).logits
        logits_cont = logits_all[:,len_prompt-1:,:][0].detach().cpu().numpy()
    tokens_cont = tokens[:,len_prompt-1:]
    ind2noise = {}
    all_gumbel_noise = []

    alphas = np.exp(logits_cont)
    logsumalphas = np.log(np.sum(alphas, axis=1))
    
    for i, w in enumerate(tokens_cont[0,1:]):
        w_int = w.detach().cpu().numpy().item()
        gumbel_noise = topdown_vectorized(alphas[i], w_int, logsumalphas[i], logits_cont[i])
        all_gumbel_noise.append(gumbel_noise)
        ind2noise[i] = (tokenizer.decode(w), gumbel_noise)
    
    all_gumbel_noise = np.array(all_gumbel_noise)
    return all_gumbel_noise

    
