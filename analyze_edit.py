import pickle
import matplotlib.pyplot as plt
import numpy as np

models = ["counterfactuals2/wiki_Meta-Llama-3-8B-Instruct->mimic_gender_llama3_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_Meta-Llama-3-8B-Instruct->honest_steering_llama3_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_Meta-Llama-3-8B->chat_llama3_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_gpt2-xl->mimic_gender_gpt2_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_gpt2-xl->GPT2-memit-louvre-rome_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals2/wiki_gpt2-xl->GPT2-memit-koalas-new_zealand_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl"]
names = ["LLaMA3-Steering-Gender", "LLaMA3-Steering-Honest", "LLaMA3-Instruct", "GPT2-XL-Steering-Gender", "GPT2-XL-MEMIT-Louvre", "GPT2-XL-MEMIT-Koalas"] #["Honest-LLama", "GPT-XL-ROME", "LLama2-Chat", "GPT2-XL-MEMIT"

models =  ["counterfactuals/wiki_Meta-Llama-3-8B-Instruct->mimic_gender_llama3_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
           "counterfactuals/wiki_gpt2-xl->mimic_gender_gpt2_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
           "counterfactuals/wiki_Meta-Llama-3-8B-Instruct->honest_steering_llama3_instruct_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals/wiki_Meta-Llama-3-8B->chat_llama3_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals/wiki_gpt2-xl->GPT2-memit-koalas-new_zealand_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl",
         "counterfactuals/wiki_gpt2-xl->GPT2-memit-louvre-rome_prompt:first_k_sents:500_prompt_first_k:5_max_new_tokens:25.pkl"]
names = ["LLaMA3-Steering-Gender", "GPT2-XL-Steering-Gender", "LLaMA3-Steering-Honest", "LLaMA3-Instruct", "GPT2-XL-MEMIT-Koalas", "GPT2-XL-MEMIT-Louvre"]

#models = ["counterfactuals3/wiki_Meta-Llama-3-8B-Instruct->mimic_gender_llama3_instruct_prompt:first_k_sents:50_prompt_first_k:5_max_new_tokens:25.pkl"]
#names = ["LLaMA3-Steering-Gender"]

name2data = {}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "cyan", "purple", "red"] #['blue', 'orange', 'green', "red", "cyan"]  # Define colors for consistency

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 6))


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

EDIT_DISTANCE=True

print(names, models)
for idx, (name, model) in enumerate(zip(names, models)):
    print(name)
    with open(model, "rb") as f:
        data = pickle.load(f)
        orig, count = data["original"], data["counter"]
        #counter = [d["counter"] for d in counter] 
        #orig = [o.split(" ") for o in original]
        #count = [c.split(" ") for c in counter]

        orig = [d["text"] for d in orig]
        print(count[0])
        count = [d["text"] for d in count]
        name2data[name] = (orig, count)

        
        diffs=[]
        #print(len(orig), len(count))
        
        for o,c in zip(orig, count):
            
            #print(o,c)
             if EDIT_DISTANCE:
                 diffs.append(levenshteinDistance(o,c)/len(c))
             else:
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
        mean_diff = np.mean(diffs)
        print(np.mean(diffs))
        plt.axvline(
            mean_diff,
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
plt.xlabel("Edit Distance (characters)", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.grid()
plt.legend(fontsize=13)
plt.savefig("edit_distance_new.pdf", dpi=800)
