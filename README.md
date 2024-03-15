# playchatglm3-6b

gemma2b/7b mistral are very unstable in classifing a joke,so the rewards model for Agents is switched to chatglm,the objective is not changed：

1、gemma2 can summarize ibdb reviews,use lenght, as rewards,to guide the model to generate shorter output

2、gemma2 or many chinese LLMs can tell jokes,use like and reply as rewards,to guide the model to generate better jokes

3、multi agents can be SFTed by other agents' positive samples and guided by self positive sample with rl. This shows how human evolved
