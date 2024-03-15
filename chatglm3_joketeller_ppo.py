from transformers import AutoTokenizer, pipeline, AutoModel
import torch
import bitsandbytes
import accelerate

base_model_it = "/DATA/jupyter/personal/THUDM/chatglm3-6b"

#init UI or a third part LLM API as a reward model to score a joke 



#test chatglm3 function
tokenizer = AutoTokenizer.from_pretrained(base_model_it, trust_remote_code=True)
model = AutoModel.from_pretrained(
    base_model_it,
    trust_remote_code=True).quantize(4).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "tell me a joke", history=[])

print(response)


#init both A B model as PEFT mode for rl

#outer loop

#inner RL loop

#load all models seperatly

#all models generate a joke seperatly,using a different random seed

#reward model give all the jokes a score seperatly

#both model A B run a PPO step,and save the generation as positive sample if reward is bigger than threshhold

#inner rl loop end

#model save checkpoint
#evaluate with reward model

#A model sfted by positive sample from B, vice versa

#model save checkpoint
#eval with reward model

#end outer loop

print("end")