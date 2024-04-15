from transformers.generation import LogitsProcessor,LogitsProcessorList
import numpy as np
from scipy.stats import gumbel_l
from arsenal.maths.rvs import TruncatedDistribution

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


class GumbelProcessor(LogitsProcessor):
    def __init__(self, precomputed_noise=None):
        self.precomputed_noise = precomputed_noise
        self.i=0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.precomputed_noise is not None:
            return scores + self.precomputed_noise[self.i]
        gumbel = np.random.gumbel(loc=0.0, scale=1.0, size=scores.shape)
        self.i+=1
        return scores + gumbel


def sample_gumbel(model, tokenizer, gumbel_processor, prompt):

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=64, logits_processor=[gumbel_processor],
                                   do_sample=False)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True)



def counterfactual_generation(model, tokenizer, sentence, vocab_size):

    tokens = tokenizer(sentence, return_tensors="pt")
    tokens = tokens.input_ids
    logits = model(tokens).logits.detach().cpu().numpy()
    all_gumbel_noise = []

    tokens = tokens[0]
    for i,w in enumerate(tokens[1:]):
        value = np.random.gumbel(loc=0.0, scale=1.0)
        logit_w = logits[0][i][w]
        gumbel_noise = []
        for j in tqdm.tqdm(range(vocab_size)):
            logit_j = logits[0][i][j]
            truncated_gumbel = TruncatedDistribution(gumbel_r, a=-100, b= value + logit_w - logit_j)
            sample = truncated_gumbel.rvs(size=1)
            gumbel_noise.append(sample.item())
        gumbel_noise[w] = value        
        all_gumbel_noise.append(gumbel_noise)  
    
    all_gumbel_noise  = np.array(all_gumbel_noise)
    processor = GumbelProcessor(precomputed_noise=torch.tensor(all_gumbel_noise))

    return sample_gumbel(model, tokenizer, processor, "The")

if __name__ == "__main__":
  
  model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
  tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B")
  model.eval()
  processor = GumbelProcessor()
  sample_gumbel(model, tokenizer, processor, "one day,")
