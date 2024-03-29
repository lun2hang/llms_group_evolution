#modify a trl demo to orgnize a group of llms to generate more positive comments to a movie, a bert based sentiments classifier acts as reward model

'''
group evolution fake code:
for i in num_evolution
    for i in num_epoch
        for j in num_llms
            train_ppo(llms[j])
            dump positive sample to llms_positive_sample[j]
        for j in num_llms
            train_sft(llms[j],llms_positive_sample[!=j])        
    replace the Bottom N llms with dupliations of the Top M llms
'''
from transformers import AutoTokenizer, pipeline, AutoModel
import torch
import bitsandbytes
import accelerate

num_evolution = 2
num_epoch = 2
num_llms = 2

#init a reward model(bert scentiment classifier) as environment to score a response

#init base model

#loop for evolution start
for i in range(num_evolution):
#loop for epoch start
    for j in range(num_epoch):
#inner loop for rl start
        for k in range(num_llms):
#PPO train: here we use a sequnential ppo training to save GPU memory
            print("RL from feedback from envrionment in evol: %d, epoch: %d, llms: %d " % (i, j, k))
#init the peft model with different adaptor for specific llms

#model[i] generate a response ,using a randomly sampled query

#reward model give the responses a score

#run a PPO step,and save the generation as positive sample if reward is bigger than threshhold

#model save checkpoint
#evaluate with reward model

#ppo train next llms
        for k in range(num_llms):
            print("SFT from other llm's expericences in evol: %d, epoch: %d, llms: %d " % (i, j, k))
#inner rl loop end

#inner sft loop start

#A model sfted by positive sample from all other models, vice versa

#model save checkpoint
#eval with reward model

#inner sft loop end



#loop for epoch end

#drop the bottom models,duplicate the top models

#loop for evolution end

print("end")