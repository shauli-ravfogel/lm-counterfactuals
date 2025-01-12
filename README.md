This repository contains the code for the paper "[Gumbel Counterfactual Generation from Language Models](https://arxiv.org/pdf/2411.07180)". In this work, we conceptualize LMs as Generalized Causal Models (GCMs), enabling us to generate true counterfactual strings from a given input string. By leveraging the Gumbel-Max trick, we separate the deterministic computations of the LM’s forward pass from the inherent randomness of the sampling process. This allows us to use hindsight sampling to identify the noise responsible for generating a specific string and reuse the same noise when generating a counterfactual string from the model, post-intervention.

To set up the environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# download models
gdown --folder https://drive.google.com/drive/folders/11PE8DxVqbfpsLhqLop71CqL6KsdeTnFf
```
Then, run ```run.py``` to re-generate the counterfactuals on the Wikipedia/Bios dataset. The notebook [example.ipynb](https://github.com/shauli-ravfogel/lm-counterfactuals/blob/main/example.ipynb) contains a minimal example for generating a counterfactual string based on an original string.

The directory ```counterfactuals``` contains the counterfactuals sentences we generated from Wikipedia and the Biosd dataset, based on several models and intervention techniques.
