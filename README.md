# playchatglm3-6b

gemma mistral对笑话等级的划分输出不稳定，因此多Agents共同学习计划在chatglm上重启，并切换到中文，继续探索多Agents相互SFT学习&分别向环境RL进化的计划不变：

1、gemma2 can summarize ibdb reviews,use lenght, as rewards,to guide the model to generate shorter output

2、gemma2 or many chinese LLMs can tell jokes,use like and reply as rewards,to guide the model to generate better jokes

3、multi agents can be SFTed by other agents' positive samples and guided by self positive sample with rl. This shows how human evolved
