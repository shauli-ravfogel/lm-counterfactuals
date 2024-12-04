import utils
import argparse
from mimic import InterventionModule, insert_intervention, insert_intervention
from transformers import AutoTokenizer
import transformers
import tqdm
import pickle
from sklearn.utils import shuffle
from transformers import pipeline
import torch
import numpy as np
from sklearn.linear_model import SGDClassifier

@torch.no_grad()
def encode(model, tokenizer, text, batch_size,layer=-1, pooling="last", max_len=128):
  encodings = []

  with torch.no_grad():
   for i in tqdm.tqdm(range(0, len(text), batch_size)):
    batch = text[i:i+batch_size]
    padded_tokens = tokenizer(batch, padding=True, return_tensors="pt", max_length=max_len, truncation=True).to("cuda")
    outputs = model(**padded_tokens, output_hidden_states=True)
    lengths = padded_tokens["attention_mask"].sum(axis=1).detach().cpu().numpy()

    hiddens = outputs.hidden_states[layer]
    hiddens = hiddens.detach()
    for h,l in zip(hiddens, lengths):
      if pooling == "last":
        h = h[l-1]
      elif pooling == "cls":
        h = h[0]
      elif pooling == "mean":
        h = h[:l].mean(axis=0)
      encodings.append(h.detach().cpu().numpy())

  return np.array(encodings)

import scipy

def get_optimal_gaussian_transport_func(source_x, target_x, reg=1e-7):
      cov_source = np.cov(source_x.T).real + reg
      cov_target = np.cov(target_x.T).real + reg

      # optimal transport

      cov_source_sqrt = matrix_squared_root(cov_source)
      cov_source_sqrt_inv = matrix_inv_squared_root(cov_source) #scipy.linalg.inv(cov_source_sqrt)

      A = cov_source_sqrt_inv @ matrix_squared_root(cov_source_sqrt @ cov_target @ cov_source_sqrt) @ cov_source_sqrt_inv
      return A

def matrix_squared_root(A):
    return scipy.linalg.sqrtm(A)


def matrix_inv_squared_root(A):

    return np.linalg.inv(matrix_squared_root(A))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")   #parser is an object of the class Argument Parser.
    #parser.add_argument("--dataset_name", type=str, default="sentence-transformers/wikipedia-en-sentences", required=False)
    parser.add_argument("--dataset_name", type=str, default="bios", required=False)
    parser.add_argument("--mimic_alpha", type=float, default=1.0,required=False)
    #parser.add_argument("--model", type=str, default="openai-community/gpt2-xl")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--tokenizer_max_len", type=int, default=16)

    args = parser.parse_args()    
    model_name = args.model.split("/")[1]
    model = utils.load_model(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, model_max_length=args.tokenizer_max_len, padding_side="right", use_fast=False,trust_remote_code=True)
    if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
    tokenizer.pad_token=tokenizer.eos_token
    
    with open("../bios_data/bios_train.pickle", "rb") as f:
        data = pickle.load(f)
    y = np.array([d["p"] for d in data])
    z = np.array([1 if d["g"] == "m" else 0 for d in data])
    texts = [d["text"] for d in data]

    y_to_keep = ["professor"]
    idx_to_keep = [i for i in range(len(y)) if y[i] in y_to_keep]
    y = y[idx_to_keep]
    z = z[idx_to_keep]
    texts = [texts[i] for i in idx_to_keep]

    num_m, num_f = sum(z), len(z) - sum(z)
    n = 10000#min(num_m, num_f)
    idx_m = [i for i in range(len(z)) if z[i] == 1]
    idx_f = [i for i in range(len(z)) if z[i] == 0]
    idx = idx_m[:n] + idx_f[:n]
    y = y[idx]
    z = z[idx]
    texts = [texts[i] for i in idx]
    texts, y, z = shuffle(texts, y, z, random_state=0)


    encodings = encode(model, tokenizer, texts[:args.n], args.bsize, layer=args.layer, pooling="mean", max_len = args.tokenizer_max_len)
    # save the texts, encodings, and labels
    with open(f"interim/encodings_{args.layer}_model={model_name}.pickle", "wb") as f:

        pickle.dump({"texts": texts[:len(encodings)], "encodings": encodings, "y": y[:len(encodings)], "z": z[:len(encodings)]}, f)

    with open(f"interim/encodings_{args.layer}_model={model_name}.pickle", "rb") as f:
     data = pickle.load(f)
     texts = data["texts"]
     encodings = data["encodings"]
     y = data["y"]
     z = data["z"]

    print(texts[0], z[0])
    
    x_train_source = encodings[z==1,:][:]
    x_train_target = encodings[z==0,:][:]
    A = get_optimal_gaussian_transport_func(x_train_source,x_train_target, reg=1e-5)
    mean_source = x_train_source.mean(axis=0)
    mean_target = x_train_target.mean(axis=0)
    #clf = SGDClassifier(loss="log_loss", max_iter=10000, tol=1e-4)
    #clf.fit(encodings, z_subset)

    intervention_module = InterventionModule(mean_source, mean_target, A, None, alpha=args.mimic_alpha)
    insert_intervention(model, model_name, intervention_module, args.layer, replace_existing=False,after_layer_norm=True)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    # generate text
    prompt = "James Richardson is an Assistant Professor at Jacksonville State University (JSU) in"
    for i in range(16):
        print(generator(prompt, max_length=75, num_return_sequences=1, do_sample=True))
        print("================================")
    with open("interim/mimic_gender_{}_layer={}.pickle".format(model_name, args.layer), "wb") as f:
        pickle.dump(intervention_module.cpu(), f)