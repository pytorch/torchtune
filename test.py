import torch

from torchtune.models.llama2 import llama2_7b
from torchtune.models.lora_llama2 import lora_llama2_7b as lora


with torch.device("cuda:0"):
    ll = lora(['q_proj', 'v_proj'])

with torch.device("cuda:1"):
    lm = llama2_7b()

lms = lm.state_dict()
lls = ll.state_dict()

k1 = list(lms.keys())
k2 = list(lls.keys())

# All keys should match up besieds k2 having 'lora'

j1 = [k for k in k2 if k not in k1]

assert all(['lora' in j for j in j1])
# Everything in k1 should be in k2
missing_in_lora = [k for k in k1 if k not in k2]
assert all([k in k2 for k in k1])

# And tensor shapes should match up
for k in k1:
    s1 = lms[k].shape
    s2 = lls[k].shape
    assert s1 == s2
