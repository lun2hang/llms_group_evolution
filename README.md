# Group evolution,next generation LLM training paradigm

0 LLMS can mutually compete with and enlighten each other to form a stronger group

1 Multi agents can be SFTed by other agents' positive samples and guided by self experiences with rl. This shows how human evolved

2 Agents can have lifetime,top Agents clone themselves,bottom Agents vanish.

3 All the mechanism realize a ecosystem obeying Theory of Evolution.

4 This training paradigm will work in all multiagents scene,Ads\Recommendation\Unmanned call center\Unmanned Vehicle\Human kind robots\Drones

5 LLMs can be deployed in different data center,which means this fit power/data distribution well and can scale up to the whole world just like what we human being do

Experiment result:
After TRL demo reaches a reward of 1.6, it begins to vibrate;
RL&SFT interleaved training gets a reward of 2.65;
Dynamic sample_keep threshhold pushes the score to 2.85;
Evolution mechanism lifts the reward at evolution4 by 0.15,reward can keep rising in 45 epochs;
To avoid the sft fail from no training data caused by a too high dynamic threshhold,we always keep the max reward sample no matter above or blew the threshhold;
Finally, all llms in the group beat reward model upper bound at 2.9x with a max reward of 2.99

These parameters will balance the training speed and final score: dynamic threshhold coefficient 1.1,threshhold 2.96,epoch 45,ppo steps/epoch 4,num_llms 3,evla_batchs 1
