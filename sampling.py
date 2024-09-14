from transformers.generation import LogitsProcessor,LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
#from arsenal.maths.rvs import TruncatedDistribution
import copy
import torch
import tqdm
import scipy
from scipy.stats import gumbel_l, gumbel_r
#from arsenal.maths.rvs import TruncatedDistribution
import transformers
from evaluate import load


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


def switch_gumbel_noise(noise, replaced_pairs, tokenizer):

    noise_new = copy.deepcopy(noise)
    for pair in replaced_pairs:
        for i in range(len(noise)):
            token_id_0 = tokenizer(pair[0])["input_ids"][0]
            token_id_1 = tokenizer(pair[1])["input_ids"][0]
            noise_new[i][0][token_id_0], noise_new[i][0][token_id_1] = noise[i][0][token_id_1], noise[i][0][token_id_0]
    return noise_new


def switch_tokens(model, tokenizer, replaced_pairs, swap_only_input=False, swap_only_output=False):
    ###
    # model: the model to be edited 
    # tokenizer: the tokenizer used to tokenize the text
    # replaced_pairs: a list of tuples of strings, each tuple contains two strings that are to be swapped
    # swap_only_input: if True, only the input embeddings are swapped
    # swap_only_output: if True, only the output embeddings are swapped
    # Note: the implementation assumes the input and output embeddings are tied in the original model
    # returns: the edited model

    # assert tied embeddings
    assert id(model.transformer.wte.weight) == id(model.lm_head.weight)

    # swap pairs of tokens
    for pair in replaced_pairs:
        token_id_0 = tokenizer(pair[0])["input_ids"][0]
        token_id_1 = tokenizer(pair[1])["input_ids"][0]
        embedding_pair_0 = model.transformer.wte.weight.data[token_id_0].clone()
        embedding_pair_1 = model.transformer.wte.weight.data[token_id_1].clone()

        if swap_only_input or swap_only_output:
            # untie the input and output embeddings
            embedding_mat = copy.deepcopy(model.transformer.wte.weight)
            model.lm_head.weight = embedding_mat.clone()
            # assert the objects are not the same anymore
            assert id(model.transformer.wte.weight) != id(model.lm_head.weight)

            if swap_only_output:
                model.lm_head.weight.data[token_id_0, :] = embedding_pair_1.clone()
                model.lm_head.weight.data[token_id_1, :] = embedding_pair_0.clone()
                continue

            elif swap_only_input:
                model.transformer.wte.weight.data[token_id_0, :] = embedding_pair_1.clone()
                model.transformer.wte.weight.data[token_id_1, :] = embedding_pair_0.clone()
                continue
        else:  # swap both input and output embeddings
            model.transformer.wte.weight.data[token_id_0, :], model.transformer.wte.weight.data[token_id_1, :] = embedding_pair_1.clone(), embedding_pair_0.clone()
            model.lm_head.weight.data[token_id_0, :], model.lm_head.weight.data[token_id_1, :] = embedding_pair_1.clone(), embedding_pair_0.clone()

    return model


    
class GumbelProcessor(LogitsProcessor):
    def __init__(self, precomputed_noise=None,noise=1, replaced_pairs=None):
        self.precomputed_noise = precomputed_noise
        self.i=0
        self.replaced_pairs=replaced_pairs
        # set np random seed

        #np.random.seed(noise)
        self.noises = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        
        if self.precomputed_noise is not None:
            out = scores + self.precomputed_noise[self.i]
            self.i += 1
            return out
        
        # gumbel = np.random.gumbel(loc=0.0, scale=1.0, size=scores.shape)
        # self.noises.append(gumbel)
        # return scores + gumbel



def sample_gumbel(model, tokenizer, gumbel_processor, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=64, logits_processor=[gumbel_processor],
                                   do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True)


def unconditional_counterfactual_generation(model, tokenizer, vocab_size):
    all_gumbel_noise = []

def counterfactual_generation_partial(model, tokenizer, sentence, vocab_size,prompt=None):

    tokens = tokenizer(sentence, return_tensors="pt")
    tokens = tokens.input_ids
    logits = model(tokens).logits.detach().cpu().numpy()
    all_gumbel_noise = []
    cdf_a = gumbel_l.cdf(-500.0)
    for i,w in enumerate(tokens[0][1:]):
        value = np.random.gumbel(loc=0.0, scale=1.0)
        logit_w = logits[0][i][w]
        gumbel_noise = []
        for j in tqdm.tqdm(range(vocab_size)):
            logit_j = logits[0][i][j]
            #truncated_gumbel = TruncatedDistribution(gumbel_r, a=-300, b= value + logit_w - logit_j)
            #sample = truncated_gumbel.rvs(size=1)
            sample = sample_from_truncated_gumbel(cdf_a, value + logit_w - logit_j, gumbel_l, j)
            gumbel_noise.append(sample)
            #gumbel_noise.append(1)
        gumbel_noise[w.detach().cpu().numpy().item()] = value        
        all_gumbel_noise.append(gumbel_noise)  
    
    return all_gumbel_noise




# def sample_from_truncated_gumbel_vectorized(cdf_a, b_array, gumbel):
#     cdf_b = gumbel.cdf(b_array)  # Compute CDF for an array of b values
#     u = []
#     for i in range(b_array.shape[0]):
#         np.random.seed(i)
#         u.append(np.random.uniform(0, 1))

#     #u = np.random.uniform(0, 1, size=b_array.shape)  # Generate uniform random values of the same shape as b_array
#     return gumbel.ppf(cdf_a + u * (cdf_b - cdf_a))  # Apply the inverse CDF to each element in the array




def counterfactual_generation(model, tokenizer, sentence, vocab_size,prompt=None):

    tokens = tokenizer(tokenizer.bos_token + sentence, return_tensors="pt")
    tokens = tokens.input_ids
    logits = model(tokens).logits.detach().cpu().numpy()
    all_gumbel_noise = []
    cdf_a = gumbel_l.cdf(-500.0)
    all_logit_diffs = []
    
    for i,w in enumerate(tokens[0][1:]):
        #np.random.seed(i)
        value = np.random.gumbel(loc=0.0, scale=1.0)
        logit_w = logits[0][i][w]
        gumbel_noise = [] 
        logit_diffs = []
        
        for j in tqdm.tqdm(range(vocab_size)):
            logit_j = logits[0][i][j]
            #np.random.seed(0)
            sample = sample_from_truncated_gumbel(cdf_a, value + logit_w - logit_j, gumbel_l)
            logit_diffs.append(logit_w - logit_j)
            gumbel_noise.append(sample)
            #gumbel_noise.append(1)
        all_logit_diffs.append(logit_diffs)
        gumbel_noise[w.detach().cpu().numpy().item()] = value 
        w_ind = w.detach().cpu().numpy().item()
        for j in range(len(gumbel_noise)):
            if j != w_ind:
                assert logits[0][i][j] + gumbel_noise[j] < logit_w + value
        print(w, "all good")
        all_gumbel_noise.append(gumbel_noise)  

    # add a bias to the EOS token to make it more likely at the end
    
    eos = tokenizer.eos_token_id
    noise = np.zeros(vocab_size)
    noise[eos] = 500.0
    all_gumbel_noise.append(noise)
    
    all_gumbel_noise  = np.array(all_gumbel_noise)
    processor = GumbelProcessor(precomputed_noise=torch.tensor(all_gumbel_noise))
    return sample_gumbel(model, tokenizer, processor, tokenizer.bos_token)#, all_logit_diffs, all_gumbel_noise, processor


def counterfactual_generation_batched(model, tokenizer, sentence, vocab_size, prompt=None):
    tokens = tokenizer(tokenizer.bos_token+sentence, return_tensors="pt").input_ids
    logits = model(tokens).logits.detach().cpu().numpy()
    all_gumbel_noise = []

    cdf_a = gumbel_l.cdf(-500.0)
    all_logit_diffs = []
    for i, w in enumerate(tokens[0][1:]):
        # Generate Gumbel noise for each token in the vocabulary
        #np.random.seed(i)
        value = np.random.gumbel(loc=0.0, scale=1.0)
        # Calculate the logits differences
        logit_diffs =  logits[0][i][w] - logits[0][i] 
        all_logit_diffs.append(logit_diffs)
        # Apply vectorized truncated Gumbel sampling to the entire vocabulary
        truncated_gumbels = sample_from_truncated_gumbel_vectorized(cdf_a, value + logit_diffs, gumbel_l)

        # Ensure the original token keeps its specific noise
        truncated_gumbels[w.detach().cpu().numpy().item()] = value 
        
        all_gumbel_noise.append(truncated_gumbels)

    # Add a bias to the EOS token to make it more likely at the end
    eos = tokenizer.eos_token_id
    noise = np.zeros(vocab_size)
    noise[eos] = 500.0
    all_gumbel_noise.append(noise)

    all_gumbel_noise = np.array(all_gumbel_noise)

    # Convert Gumbel noise to a tensor
    processor = GumbelProcessor(precomputed_noise=torch.tensor(all_gumbel_noise))

    #first_word = sentence.split(" ")[0]
    #prompt = tokenizer.bos_token if prompt is None else prompt
    return sample_gumbel(model, tokenizer, processor, tokenizer.bos_token)#, all_logit_diffs, all_gumbel_noise, processor


def sample_from_truncated_gumbel(cdf_a,b,gumbel,seed=None):
    cdf_b = gumbel.cdf(b)
    #np.random.seed(0) if seed is None else np.random.seed(seed)
    u = np.random.uniform(0, 1)
    if cdf_b < cdf_a:
        cdf_a, cdf_b = cdf_b, cdf_a
    return gumbel.ppf(cdf_a + u * (cdf_b - cdf_a))
    
def sample_from_truncated_gumbel_vectorized(cdf_a, b):
    cdf_a = np.zeros_like(b)
    #return np.array([sample_from_truncated_gumbel(cdf_a[i], b[i], gumbel_l) for i in range(len(b))])
    
    # Calculate the CDF for b and scale u accordingly for truncation
    cdf_b = gumbel_l.cdf(b)  # This is the scalar or array CDF for b
    u=[]
    for i in range(b.shape[0]):
        #np.random.seed(i)
        u.append(np.random.uniform(0, 1))
    u = np.array(u)

    cdf_a_copy = np.zeros_like(b)
    cdf_b = np.where(cdf_b > 0., cdf_b, -cdf_b)
            
    # Ensure the operation is element-wise (u * (cdf_b - cdf_a))
    cdf_u = cdf_a + u * (cdf_b - cdf_a)  # Element-wise operation between scalars or arrays
    
    # Ensure the CDF is bounded between 0 and 1 (floating-point precision issues might cause problems)
    cdf_u = np.clip(cdf_u, 0, 1)
    
    # Apply inverse CDF (ppf) to the result, element-wise
    return gumbel_l.ppf(cdf_u)


def counterfactual_generation_vectorized(model, tokenizer, sentence, vocab_size, prompt=None):
    tokens = tokenizer(tokenizer.bos_token + sentence, return_tensors="pt").input_ids
    logits = model(tokens).logits.detach().cpu().numpy()

    # Pre-calculate the CDF and Gumbel noise for all tokens
    cdf_a = gumbel_l.cdf(np.ones(1)*(-500.0))
    all_logit_diffs = []
    all_gumbel_noise = []

    # Generate a uniform sample U for Gumbel sampling
    #np.random.seed(0)
    uniform_samples = np.random.uniform(0, 1, size=(len(tokens[0]) - 1, vocab_size))

    for i, w in enumerate(tokens[0][1:]):
        # Get logits for current word and all vocab
        logit_w = logits[0][i, w]
        logit_diffs = logit_w - logits[0][i]  # Corrected: logit_w - logit_j for all vocab

        # Generate gumbel noise for the current word
        #np.random.seed(i)
        value = np.random.gumbel(loc=0.0, scale=1.0)

        # Calculate the sample from truncated gumbel for all vocabulary items
        gumbel_noise = sample_from_truncated_gumbel_vectorized(cdf_a, value + logit_diffs)
        # cdf_b = gumbel_l.cdf(value + logit_diffs)
        # cdf_a = np.zeros_like(cdf_b)
        # u = np.random.rand(cdf_b.shape[0])
        # gumbel_noise = gumbel_l.ppf(cdf_a + u * (cdf_b - cdf_a))
        print(w)
        # Replace the noise for the actual token with its own Gumbel value
        gumbel_noise[w.detach().cpu().numpy().item()] = value

        all_gumbel_noise.append(gumbel_noise)
        all_logit_diffs.append(logit_diffs)

    # Add EOS token bias
    eos = tokenizer.eos_token_id
    noise = np.zeros(vocab_size)
    noise[eos] = 500.0
    all_gumbel_noise.append(noise)

    # Convert results to NumPy arrays
    all_gumbel_noise = np.array(all_gumbel_noise)
    all_logit_diffs = np.array(all_logit_diffs)
    # Convert Gumbel noise to a tensor
    processor = GumbelProcessor(precomputed_noise=torch.tensor(all_gumbel_noise))

    first_word = sentence.split(" ")[0]
    prompt = tokenizer.bos_token if prompt is None else prompt
    return sample_gumbel(model, tokenizer, processor, tokenizer.bos_token)#, all_logit_diffs, all_gumbel_noise, processor


def get_perp(model, tokenized_prompt):

  with torch.no_grad():
        outputs = model(tokenized_prompt)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tokenized_prompt.view(-1), reduction="none")
    
  perp = torch.exp(loss)
  return loss

if __name__ == "__main__":
  #model_name = "google/gemma-2b-it"
  model_name = "openai-community/gpt2-large"
  swapped_pairs = [("her", "him"), (" her", " him"), ("she", "he"), (" she", " he")]         

  model = AutoModelForCausalLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model.eval()
  processor = GumbelProcessor()
  sample = sample_gumbel(model, tokenizer, processor, "one rainy day, I saw him")
  print("=============================")
  print("Single sample:", sample[0])
  print("=============================")

  # create an edited model

  edited_model = copy.deepcopy(model)
  edited_model = switch_tokens(edited_model, tokenizer, swapped_pairs)


  # sample from the edited model
  generation_pipeline = transformers.pipeline("text-generation", model=edited_model, tokenizer=tokenizer, device=0)
  # generate
  print("Sample from edited model:", generation_pipeline("I thought I identified John, and when I saw", max_length=100, num_return_sequences=1, do_sample=False, truncation=True)[0]["generated_text"])
  print("=============================")

  # counterfactual generation
  print("Generating counterfactual pair of continuations")
  gumbel_processor = GumbelProcessor(noise=2)
  prompt = "I saw him"
  tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"]
  prompt_tokens = [tokenizer.decode(t) for t in tokenized_prompt[0]]
  print(prompt_tokens)
  print(sample_gumbel(model.cpu(), tokenizer, gumbel_processor, prompt))

  noises_first = gumbel_processor.noises
  noises_first = switch_gumbel_noise(noises_first, swapped_pairs, tokenizer)
  gumbel_processor = GumbelProcessor(noise=2, precomputed_noise=noises_first)
  prompt = "I saw her"
  tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"]
  prompt_tokens = [tokenizer.decode(t) for t in tokenized_prompt[0]]
  print(sample_gumbel(edited_model.cpu(), tokenizer, gumbel_processor, prompt))
  print(prompt_tokens)

  noises_second = gumbel_processor.noises
  
  assert np.allclose(noises_first, noises_second)

  # generate a pair of unconditional counterfactuals

  # Likelihood test

  print("=============================")
  print("Likelihood test")
  prompt = "she approached me, and I saw her"
  tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"]
  perp = get_perp(model, tokenized_prompt)
  print("perplexity of the original model on `{}`".format(prompt), perp)
  
  prompt = "he approached me, and I saw him"
  tokenized_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"]
  perp = get_perp(edited_model.cpu(), tokenized_prompt)
  print("perplexity of edited model on `{}`".format(prompt), perp)
