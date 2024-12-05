import utils
import argparse
from mimic import InterventionModule, insert_intervention, insert_intervention
from transformers import AutoTokenizer
import transformers
import tqdm
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")   #parser is an object of the class Argument Parser.
    #parser.add_argument("--dataset_name", type=str, default="sentence-transformers/wikipedia-en-sentences", required=False)
    parser.add_argument("--dataset_name", type=str, default="bios", required=False)
    parser.add_argument("--bios_zs_to_keep", type=list, default=[1], required=False)
    parser.add_argument("--bios_ys_to_keep", type=list, default=["professor"], required=False)
    parser.add_argument("--num_sents", type=int, default=500,required=False)
    parser.add_argument("--prompt", type=str, default="first_k",required=False)
    parser.add_argument("--prompt_first_k", type=int, default=7,required=False)
    parser.add_argument("--max_new_tokens", type=int, default=40,required=False)
    parser.add_argument("--num_counterfactuals", type=int, default=1,required=False)
    parser.add_argument("--models", type=list, default=[
         ("openai-community/gpt2-xl", "mimic_gender_gpt2_instruct"),
         ("meta-llama/Meta-Llama-3-8B-Instruct", "mimic_gender_llama3_instruct"),
         ("meta-llama/Meta-Llama-3-8B-Instruct", "honest_steering_llama3_instruct"),
         ("meta-llama/Meta-Llama-3-8B", "chat_llama3"),
         ("openai-community/gpt2-xl", "rome_louvre_gpt2_xl"),
         ("openai-community/gpt2-xl", "GPT2-memit-louvre-rome"),
         ("openai-community/gpt2-xl", "GPT2-memit-koalas-new_zealand")
            ], required=False)
    args = parser.parse_args()    
    # load data
    
    bios_args = {"zs_to_keep": args.bios_zs_to_keep, "ys_to_keep": args.bios_ys_to_keep}
    sents = utils.load_sents_dataset(args.dataset_name, bios_args)[:args.num_sents]
    
    
    for (orig, intervention_type) in args.models: 
        utils.set_seed(0)
        
        # load model
        
        original_model = utils.load_model(orig)
        counterfactual_model = utils.get_counterfactual_model(intervention_type)
        tokenizer = transformers.AutoTokenizer.from_pretrained(orig, model_max_length=512, padding_side="right", use_fast=False,trust_remote_code=True)
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        
        all_sents, all_outputs = [], []
        for sentence in tqdm.tqdm(sents):

            # construct prompt
            
            if args.prompt == "first_k":
                prompt = tokenizer.bos_token + " ".join(sentence.split(" ")[:args.prompt_first_k])
            else:
                prompt = tokenzier.bos_token

            # get counterfactual
            counterfactuals = []
            original_continuation_tokens, original_continuation = utils.get_continuation(original_model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, return_only_continuation=True,num_beams=1, do_sample=True, token_healing=True)
            for l in range(args.num_counterfactuals):
                count_tokens, count_text = utils.get_counterfactual_output(counterfactual_model, original_model, tokenizer, prompt, original_continuation, args.max_new_tokens)
                counterfactuals.append({"tokens": count_tokens, "text": count_text})
            all_outputs.append(counterfactuals)
            orig_str = prompt.replace(tokenizer.bos_token,"")+original_continuation
            orig_str_tokens = tokenizer.encode(orig_str, return_tensors="pt", add_special_tokens=False).detach().cpu().numpy()[0]
            all_sents.append({"tokens": orig_str_tokens, "text": orig_str})

            print("Original: {}\n--------------------\nCounterfactual: {}".format(orig_str, count_text))
            print("==================================")
            dataset_name = "wiki" if "wiki" in args.dataset_name else "bios"
            fname = f"counterfactuals/{dataset_name}_{orig.split("/")[1]}->{intervention_type}_prompt:{args.prompt}_sents:{args.num_sents}_prompt_first_k:{args.prompt_first_k}_max_new_tokens:{args.max_new_tokens}.pkl"
            with open(fname, "wb") as f:
                pickle.dump({"original": all_sents, "counter": all_outputs}, f)
