# Adversarial Multi-Agent Reinforcement Learning

This repository is for our Final Project in Network Multi-Agent Systems. It is based on the concept of Q-Learning with 2 implementations. A basic Q-Learning algorithm and a Deep Q Network (DQN) Implementation. In addition to this, we create a custom 10x10 grid environment to train our agent. The Q-Learning algorithm and environment have foundations from the [Reinforcement Learning Series by Sentdex](https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7), while for DQN we got our foundational code from ChatGPT. 

## What is unique about this project

In this project we wanted to see how performance would be impacted by decentralized training and execution. What we mean by this is that we have 3 Agents, an Attacker, Defender and a Target. For the current case, the Target is stationary and predefined. However, the Attacker and Defender are trained individually and have rewards inverse to the distance from their goal/opponent. For example, the Attacker will gain a larger reward as it gets close to the Target but will gain a larger negative reward as it gets closer to the defender. 

## Future possible work?

If we wanted to increase our performance for either model we could do centralized training and decentralized execution. We could also give more information to the agent or even learn from a CNN rather than just Dense layers. 

## What are some results?

Due to the nature of our problem, we expect the results to roughly be 50/50 for if the Attacker wins or loses. It would also make since for the Attacker to have a slightly higher win percentage due to the fact that the Target is not moving, while the Defender will be trying to catch the attacker. What we see from our results is that we were correct in our assumptions and can see in the figure below shows that the Attacker will win ~65% and lose ~35%. 

![](Images/boxGraph.png)