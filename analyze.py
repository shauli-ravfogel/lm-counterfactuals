import pickle
import matplotlib.pyplot as plt

models = ["Meta-Llama-3-8B-Instruct_honest_llama3_8B_instruct_prompt:True.pkl",
         "gpt2-xl_rome-edits-louvre-rome_prompt:True.pkl",
         "Llama-2-7b-hf_Llama-2-7b-chat-hf_prompt:True.pkl"]

names = ["Honest-LLama", "GPT-XL-ROME", "LLama2-Chat"]


name2data = {}

for name,model in zip(names, models):
    with open(model, "rb") as f:
        data = pickle.load(f)
        original, counter = data["original"], data["counter"]
        orig = [o.split(" ") for o in original]
        count = [c.split(" ") for c in counter]
        name2data[name] = (orig, count)

        
        diffs=[]
        #print(len(orig), len(count))
        for o,c in zip(orig, count):
            #print(o,c)
            i=0
            for oo,cc in zip(o,c):
              print(cc,oo)
              if cc != oo:
                diffs.append(i/len(oo))
                break
              i+=1
        print(diffs)
        plt.hist(diffs, density=False, bins=15, alpha=0.5, label = name)
plt.legend()
plt.savefig("common_prefix.png", dpi=600)
