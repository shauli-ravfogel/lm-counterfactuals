from datasets import load_dataset
from transformers.generation.logits_process import LogitsProcessor,LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
from scipy.stats import gumbel_l
#from arsenal.maths.rvs import TruncatedDistribution
import copy
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
import tqdm
import scipy

from scipy.stats import gumbel_l, gumbel_r
#from arsenal.maths.rvs import TruncatedDistribution
import transformers
from evaluate import load
import pickle


from collections import Counter, defaultdict
import ot
import sk2torch
from sklearn.linear_model import SGDClassifier
# import nn
import torch.nn as nn
import copy
import pyreft
from sampling import generate_with_logits, counterfactual_generation_vectorized2
import tqdm
import os
from torch import Tensor
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import pyvene as pv
from transformers import pipeline

os.environ['HF_HOME'] = "/cluster/scratch/sravfogel/hf"

def direction_ablation_hook(
    activation,
    hook,
    direction
):
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj
    
if __name__ == '__main__':
    ds = load_dataset("sentence-transformers/wikipedia-en-sentences")
    num_sents = 500
    models = [ ("openai-community/gpt2-xl", "jas-ho/rome-edits-louvre-rome"),
              ("meta-llama/Meta-Llama-3-8B-Instruct", "jujipotle/honest_llama3_8B_instruct"),
             ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"),
            ]
    
    add_prompt=True
    sents = ds["train"]["sentence"][:num_sents]
    device1,device2 = "cuda:0", "cuda:1"

    fwd_hooks = None
    for orig, counter in models: 
        original_model = transformers.AutoModelForCausalLM.from_pretrained(
            orig, device_map="auto", torch_dtype=torch.float32,trust_remote_code=True)
        counterfactual_model = transformers.AutoModelForCausalLM.from_pretrained(
                counter if (not 'intervenable' in counter and not "refusal" in counter) else orig, device_map="auto",torch_dtype=torch.float32,trust_remote_code=True)
        if 'intervenable' in counter:
            pv_model = pv.IntervenableModel.load(
            counter, 
            counterfactual_model,
        )

        if "refusal" in counter:
            refusal_dir = torch.load("directions/direction_qwen_refusal.pt").to(counterfactual_model.device)
            N_INST_TEST = 32
            intervention_dir = refusal_dir
            intervention_layers = list(range(counterfactual_model.cfg.n_layers)) # all layers
            hook_fn = functools.partial(direction_ablation_hook,direction=intervention_dir)
            fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

        for param in original_model.parameters():
            param.data = param.data.to(torch.float32)
        for param in counterfactual_model.parameters():
            param.data = param.data.to(torch.float32)


        tokenizer = transformers.AutoTokenizer.from_pretrained(
            orig, model_max_length=512, 
            padding_side="right", use_fast=False,trust_remote_code=True)
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            
        vocab_size = len(tokenizer.get_vocab())

        prompt = tokenizer.bos_token
        if add_prompt:
            prompt = tokenizer.bos_token + "{}"
        conts = []

        generation_config = GenerationConfig(
                    max_new_tokens=30,
                    token_healing=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        
        for sentence in tqdm.tqdm(sents):
            original_continuation = sentence
            full_prompt = prompt if not add_prompt else prompt.format(" ".join(sentence.split(" ")[:5]))
            noise, ind2noise,logits = counterfactual_generation_vectorized2(original_model, tokenizer, full_prompt,original_continuation, vocab_size)
            #out, cont = generate_with_logits(counterfactual_model, tokenizer, full_prompt, max_new_tokens=30, noise=noise,
            #                                            first_idx = None,fwd_hooks=fwd_hooks)
            processor = GumbelProcessor(torch.tensor(noise).to(original_model.device))
            tokens_prompt = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False).to(original_model.device)
            text = original_model.generate(tokens_prompt, logits_processor=[processor], do_sample=False, generation_config=generation_config)
            text = tokenizer.decode(text.detach().cpu().numpy()[0], skip_special_tokens=True)
            conts.append(text)
            print("Generated countefactual: \n\t\t{} For sentence:\n\t\t{}".format(out,sentence))
            #counterfactual_model.to(device2)
            #original_model.to(device1)
            
            with open(f"{orig.split("/")[1]}_{counter.split("/")[1]}_prompt:{add_prompt}.pkl", "wb") as f:
                pickle.dump({"original": sents, "counter": conts}, f)
