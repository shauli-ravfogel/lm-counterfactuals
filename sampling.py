from transformers.generation import LogitsProcessor,LogitsProcessorList
import numpy as np

class GumbelProcessor(LogitsProcessor):
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        gumbel = np.random.gumbel(loc=0.0, scale=1.0, size=scores.shape)
        return scores + gumbel


def sample_gumbel(model, tokenizer, gumbel_processor, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=64, logits_processor=[gumbel_processor],
                                   do_sample=False)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

def counterfactual_generation(model, tokenizer, gumbel_processor, sentence, vocab_size):

    tokens = tokenizer(sentence, return_tensors="pt")
    tokens = tokens.input_ids

    # use the algorithm from https://timvieira.github.io/blog/post/2020/06/30/generating-truncated-random-variates/ 
    # to generate truncated random variates 

    all_gumbel_noise = []
    for i,w in enumerate(tokens[0]):
        gumbel_noise = np.zeros(vocab_size)
        gumbel_noise[w] = np.random.gumbel(loc=0.0, scale=1.0)

        # fill all over tokens with indepndent samples from the truncated gumbel distribution
        # let wn be the specific w from above, then fo each other w!=wn,
        # we samples U(w) for each token w from the truncated density p(U(w)|U(wn) + pi(wn) >= U(wn) + pi(wn))

        raise NotImplementedError("This is not implemented yet")

if __name__ == "__main__":
  
  model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
  tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B")
  model.eval()
  processor = GumbelProcessor()
  sample_gumbel(model, tokenizer, processor, "one day,")
