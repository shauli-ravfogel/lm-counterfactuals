import pickle
import matplotlib.pyplot as plt
import numpy as np

models = ["counterfactuals2/wiki_Meta-Llama-3-8B-Instruct->mimic_gender_llama3_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_Meta-Llama-3-8B-Instruct->honest_steering_llama3_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
          #"counterfactuals2/wiki_gpt2-xl->rome_louvre_gpt2_xl_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_Meta-Llama-3-8B->chat_llama3_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_gpt2-xl->mimic_gender_gpt2_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_gpt2-xl->GPT2-memit-louvre-rome_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_gpt2-xl->GPT2-memit-koalas-new_zealand_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl"]
         #]#,
         #"gpt2-xl_rome-edits-louvre-rome_prompt:True.pkl",
         #"Llama-2-7b-hf_Llama-2-7b-chat-hf_prompt:True.pkl"]
names = ["LLaMA3-Steering-Gender", "LLaMA3-Steering-Honest", "LLaMA3-Instruct", "GPT2-XL-Steering-Gender", "GPT2-XL-MEMIT-Louvre", "GPT2-XL-MEMIT-Koalas"] #["Honest-LLama", "GPT-XL-ROME", "LLama2-Chat", "GPT2-XL-MEMIT"]

name2data = {}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "cyan", "purple", "red"] #['blue', 'orange', 'green', "red", "cyan"]  # Define colors for consistency

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 6))
for idx, (name, model) in enumerate(zip(names, models)):
    print(name)
    with open(model, "rb") as f:
        data = pickle.load(f)
        orig, count = data["original"], data["counter"]
        #counter = [d["counter"] for d in counter] 
        #orig = [o.split(" ") for o in original]
        #count = [c.split(" ") for c in counter]

        orig = [d["tokens"] for d in orig]
        count = [d["tokens"][1:] for d in count]
        name2data[name] = (orig, count)

        
        diffs=[]
        #print(len(orig), len(count))
        for o,c in zip(orig, count):
            
            #print(o,c)
            i=0
            for oo,cc in zip(o,c):
              #print("try", cc,oo)
              if cc != oo:
                #print(i, len(oo))
                diffs.append(i/len(o))
                break
              i+=1
        #print(diffs)


        plt.hist(
            diffs,
            density=False,
            bins=15,
            alpha=0.5,
            label=name,
            color=colors[idx]
        )

        # Calculate and plot median
        median_diff = np.median(diffs)
        plt.axvline(
            median_diff,
            color=colors[idx],
            linestyle='dashed',
            linewidth=2
        )
        """
        plt.text(
            median_diff,
            plt.ylim()[1]*0.9 - idx*plt.ylim()[1]*0.08,  # Adjust y-position for each label
            f'Median {name}: {median_diff:.2f}',
            rotation=0,
            color=colors[idx],
            verticalalignment='top',
            horizontalalignment='center',
            fontsize=20,  # Increase font size of median labels
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
        """
plt.xlabel("Normalized Length of Longest Common Prefix", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid()
plt.legend(fontsize=13)
plt.savefig("common_prefix.pdf", dpi=800)



#### cosine sim
#### 


import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = ['query: how much protein should a female eat',
               'query: summit define',
               "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
               "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = AutoModel.from_pretrained('intfloat/e5-base-v2')

for idx, (name, model_path) in enumerate(zip(names, models)):
    print(name)
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        orig, count = data["original"], data["counter"]
        original = [d["text"] for d in orig]
        counter = [d["text"] for d in count]
        with torch.no_grad():
            batch_dict_original = tokenizer(original, max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs_original = model(**batch_dict_original)
            embeddings_original = average_pool(outputs_original.last_hidden_state, batch_dict_original['attention_mask'])
            
            batch_dict_counter = tokenizer(counter, max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs_counter = model(**batch_dict_counter)
            embeddings_counter = average_pool(outputs_counter.last_hidden_state, batch_dict_counter['attention_mask'])

            print(name, np.diag(cosine_similarity(embeddings_original,embeddings_counter)).mean())
