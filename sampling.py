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
            
def sample_from_truncated_gumbel_vectorized(a, b):
    b_copy = b.copy()
    a_copy = a.copy()
    b = np.where(b > a, b, a_copy)
    a = np.where(b_copy > a, a, b_copy)
    cdf_a = gumbel_l.cdf(a)
    cdf_b = gumbel_l.cdf(b)
    
    # Calculate the CDF for b and scale u accordingly for truncation
    
    u=[]
    for i in range(b.shape[0]):
        #np.random.seed(i)
        u.append(np.random.uniform(0, 1))
    u = np.array(u)

    # Ensure the operation is element-wise (u * (cdf_b - cdf_a))
    cdf_u = cdf_a + u * (cdf_b - cdf_a)  # Element-wise operation between scalars or arrays
    
    # Apply inverse CDF (ppf) to the result, element-wise
    return gumbel_l.ppf(cdf_u)


def truncated_gumbel_vectorized(alpha, truncation):
    """
    Vectorized version of truncated Gumbel sampling.
    """
    gumbel = np.random.gumbel(size=len(alpha)) + np.log(alpha)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))

def topdown_vectorized(alphas, k):
    """
    Vectorized version of topdown function.
    """
    topgumbel = np.random.gumbel() + np.log(np.sum(alphas))
    # Create a mask for the k-th index
    mask = np.arange(len(alphas)) == k
    gumbels = np.where(
        mask,  # If mask is True
        topgumbel,  # Assign topgumbel
        truncated_gumbel_vectorized(alphas, topgumbel)  # Else truncated_gumbel
    )
    # we calculate the gumbels with location [logit1, logits2, ..., logitn]. We need to re-center them.
    gumbels -= np.log(alphas) 
    return gumbels


def counterfactual_generation_vectorized(model, tokenizer, prompt, sentence):
    vocab_size = len(tokenizer.get_vocab())
    tokens_sentence = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=False).to(model.device)
    tokens_prompt = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    tokens = torch.cat((tokens_prompt, tokens_sentence), dim=1)
    len_prompt = len(tokens_prompt[0])

    with torch.no_grad():
        logits_all = model(tokens).logits
        logits_cont = logits_all[:,len_prompt-1:,:][0].detach().cpu().numpy()
    tokens_cont = tokens[:,len_prompt-1:]
    as_vec = np.ones(1)*(-25.0)
    uniform_samples = np.random.uniform(0, 1, size=(len(tokens[0]) - 1, vocab_size))
    ind2noise = {}
    all_gumbel_noise = []
    
    for i, w in enumerate(tokens_cont[0,1:]):
        w_int = w.detach().cpu().numpy().item()
        exp_logits = np.exp(logits_cont[i])
        gumbel_noise = topdown_vectorized(exp_logits, w_int)
        all_gumbel_noise.append(gumbel_noise)
        ind2noise[i] = (tokenizer.decode(w), gumbel_noise)
    
    all_gumbel_noise = np.array(all_gumbel_noise)
    return all_gumbel_noise

    
def counterfactual_generation_vectorized_old(model, tokenizer, prompt, sentence):
    vocab_size = len(tokenizer.get_vocab())
    tokens_sentence = tokenizer.encode(sentence, return_tensors="pt", add_special_tokens=False).to(model.device)
    tokens_prompt = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    tokens = torch.cat((tokens_prompt, tokens_sentence), dim=1)
    len_prompt = len(tokens_prompt[0])

    with torch.no_grad():
        logits_all = model(tokens).logits
        logits_cont = logits_all[:,len_prompt-1:,:][0].detach().cpu().numpy()
    tokens_cont = tokens[:,len_prompt-1:]
    as_vec = np.ones(1)*(-25.0)
    uniform_samples = np.random.uniform(0, 1, size=(len(tokens[0]) - 1, vocab_size))
    ind2noise = {}
    all_gumbel_noise = []
    
    for i, w in enumerate(tokens_cont[0,1:]):
        logit_w = logits_cont[i, w]
        logit_diffs = logit_w - logits_cont[i]  

        # Generate gumbel noise for the current word
        value = np.random.gumbel(loc=logsumexp(logits_cont[i]), scale=1.0)

        # Calculate the sample from truncated gumbel for all vocabulary items
        gumbel_noise = sample_from_truncated_gumbel_vectorized(as_vec, value + logit_diffs)
        gumbel_noise[w.detach().cpu().numpy().item()] = value
        w_ind = w.detach().cpu().numpy().item()

        all_gumbel_noise.append(gumbel_noise)
        ind2noise[i] = (tokenizer.decode(w), gumbel_noise)
    
    all_gumbel_noise = np.array(all_gumbel_noise)
    return all_gumbel_noise
