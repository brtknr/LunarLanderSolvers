# Lunar Lander Solvers

Different implementations of [LunarLander-v2] solvers in one place for evaluating the strengths and weaknesses of each approach.

## [Keras Lunar Lander](KerasLunarLander/)



## [Tensorflow Lunar Lander](TFLunarLander/)



## [Vanilla Lunar Lander](VanillaLunarLander/)

This was adapted from a [vanilla policy gradient][awjuliani-vanilla-policy-gradient] based CartPole-v0 solver which originally only had 2 outputs. While the example was a good way to understand how policy gradients work, the cost function and the recall method did not generalise all that well for LunarLander-v2.

## [Scripted Lunar Lander](ScriptedLunarLander/)

Does the job but it is not optimal, takes a long time to solve and does not generalise too well to other environments. Taken from [here][PIDAgent].

[LunarLander-v2]: https://gym.openai.com/envs/LunarLander-v2
[awjuliani-vanilla-policy-gradient]: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
[PIDAgent]: https://gym.openai.com/evaluations/eval_VaFd2xR36BOZ3INS70A