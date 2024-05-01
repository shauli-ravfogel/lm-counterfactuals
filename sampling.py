from transformers.generation import LogitsProcessor,LogitsProcessorList
import numpy as np
from scipy.stats import gumbel_l
from arsenal.maths.rvs import TruncatedDistribution
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NoiseLogger(object):

    def __init__(self, eos):
        self.eos = eos
        self.noise = []

    def __call__(self, noise: torch.FloatTensor):
        self.noise.append(noise)

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, item):
        return self.noise[item]

    def get_noise(self):
        return self.noise

    def zero_out(self):
        self.noise = []

def switch_tokens(model, tokenizer, replaced_pairs, swap_only_input=False, swap_only_output=False):
    ###
    # model: the model to be edited
    # tokenizer: the tokenizer used to tokenize the text
    # replaced_pairs: a list of tuples of strings, each tuple contains two strings that are to be swapped
    # swap_only_input: if True, only the input embeddings are swapped
    # swap_only_output: if True, only the output embeddings are swapped
    # Note: the implemention assumes the input and output embeddings are tied in the original model
    # returns: the edited model

    # assert tied emebdidngs

    assert id(model.transformer.wte.weight) == id(model.lm_head.weight)

    # sawp pairs of tokens

    for pair in replaced_pairs:
        token_id_0 = tokenizer(pair[0])["input_ids"][0]
        token_id_1 = tokenizer(pair[1])["input_ids"][0]
        embedding_pair_0 = model.transformer.wte.weight.data[token_id_0].clone()
        embedding_pair_1 = model.transformer.wte.weight.data[token_id_1].clone()

        if swap_only_input or swap_only_output:
            # untie the input and output embeddings
            embedding_mat = copy.deepcopy(model.transformer.wte.weight)
            model.lm_head.weight = embedding_mat
            # assert the objects are not the same anymore
            assert id(model.transformer.wte.weight) != id(model.lm_head.weight)

            if swap_only_output:
                model.lm_head.weight.data[token_id_0, :] = embedding_pair_1
                model.lm_head.weight.data[token_id_1, :] = embedding_pair_0
                continue

            elif swap_only_input:
                model.transformer.wte.weight.data[token_id_0, :] = embedding_pair_1
                model.transformer.wte.weight.data[token_id_1, :] = embedding_pair_0
                continue
        else: # swap both input and output embeddings
            model.transformer.wte.weight.data[token_id_0,:], model.transformer.wte.weight.data[token_id_1,:] = embedding_pair_1, embedding_pair_0
    
    return model
    
class GumbelProcessor(LogitsProcessor):
    def __init__(self, precomputed_noise=None):
        self.precomputed_noise = precomputed_noise
        self.i=0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.i+=1
        if self.precomputed_noise is not None:
            return scores + self.precomputed_noise[self.i-1]
            
        gumbel = np.random.gumbel(loc=0.0, scale=1.0, size=scores.shape)
        return scores + gumbel

def sample_gumbel(model, tokenizer, gumbel_processor, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=64, logits_processor=[gumbel_processor],
                                   do_sample=False)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

def sample_from_truncated_gumbel(cdf_a,b,gumbel):
    cdf_b = gumbel.cdf(b)
    u = np.random.uniform(0, 1)
    return gumbel.ppf(cdf_a + u * (cdf_b - cdf_a))

def counterfactual_generation(model, tokenizer, sentence, vocab_size):

    tokens = tokenizer(sentence, return_tensors="pt")
    tokens = tokens.input_ids
    logits = model(tokens).logits.detach().cpu().numpy()
    all_gumbel_noise = []
    cdf_a = gumbel_r.cdf(-500.0)
    for i,w in enumerate(tokens[0][1:]):
        value = np.random.gumbel(loc=0.0, scale=1.0)
        logit_w = logits[0][i][w]
        gumbel_noise = []
        for j in tqdm.tqdm(range(vocab_size)):
            logit_j = logits[0][i][j]
            #truncated_gumbel = TruncatedDistribution(gumbel_r, a=-300, b= value + logit_w - logit_j)
            #sample = truncated_gumbel.rvs(size=1)
            sample = sample_from_truncated_gumbel(cdf_a, value + logit_w - logit_j, gumbel_r)
            gumbel_noise.append(sample)
            #gumbel_noise.append(1)
        gumbel_noise[w.detach().cpu().numpy().item()] = value        
        all_gumbel_noise.append(gumbel_noise)  

    # add a bias to the EOS token to make it more likely at the end
    
    eos = tokenizer.eos_token_id
    noise = np.zeros(vocab_size)
    noise[eos] = 500.0
    all_gumbel_noise.append(noise)
    
    all_gumbel_noise  = np.array(all_gumbel_noise)
    processor = GumbelProcessor(precomputed_noise=torch.tensor(all_gumbel_noise))

    return sample_gumbel(model, tokenizer, processor, "The")

if __name__ == "__main__":
  
  model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
  tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
  model.eval()
  processor = GumbelProcessor()
  sample_gumbel(model, tokenizer, processor, "one day,")
