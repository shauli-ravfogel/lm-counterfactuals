from transformers.generation import LogitsProcessor,LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.stats import gumbel_l
from arsenal.maths.rvs import TruncatedDistribution
import copy
import torch
import tqdm
import scipy
from scipy.stats import gumbel_l, gumbel_r
from arsenal.maths.rvs import TruncatedDistribution
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

        np.random.seed(noise)
        self.noises = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.i += 1
        if self.precomputed_noise is not None:
            return scores + self.precomputed_noise[self.i - 1]
        
        gumbel = np.random.gumbel(loc=0.0, scale=1.0, size=scores.shape)
        self.noises.append(gumbel)
        return scores + gumbel


def sample_from_truncated_gumbel(cdf_a,b,gumbel):
    cdf_b = gumbel.cdf(b)
    u = np.random.uniform(0, 1)
    return gumbel.ppf(cdf_a + u * (cdf_b - cdf_a))

def sample_gumbel(model, tokenizer, gumbel_processor, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=64, logits_processor=[gumbel_processor],
                                   do_sample=False, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True)


def unconditional_counterfactual_generation(model, tokenizer, vocab_size):
    all_gumbel_noise = []

def counterfactual_generation(model, tokenizer, sentence, vocab_size,prompt=None):

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
    first_word = sentence.split(" ")[0]
    prompt = first_word if prompt is None else prompt
    return sample_gumbel(model, tokenizer, processor, prompt)


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
