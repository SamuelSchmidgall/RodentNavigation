***https://arxiv.org/pdf/1604.03640.pdf
https://openreview.net/pdf?id=rylU4mtUIS

# RatNavigation
Hierarchical rat navigation reinforcement learning project

**Environment Description:**
We're going to create a large 3D maze generator, which places the rodent at one corner of the maze and allows it to keep exploring until it has either - fallen or taken too long to solve. Once it fails, we place it back at the beginning of the maze (or place it upright?), and then once it has suceeded we wipe its neural memory and place it back at the beginning of a new maze. We can consider other tasks as well, ideally ones that will compliment this and enhance its neural mapping capabilities (like a simple obstacle avoidance task...).

Final Decisions:
**Visual Component Architecture:** 
Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs
https://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns


### Notes:
- 'Sense of smell' providing loose directions toward maze goal or do a sparse reward for solving maze period...
- Information Bottleneck visual inputs? -> https://arxiv.org/pdf/2002.01428v1.pdf

## Papers to consider:


**Rat Maze Behavior**
http://www.ratbehavior.org/RatsAndMazes.htm


### General:

Bio-Plausible Gradient Approx: https://arxiv.org/pdf/1608.05343.pdf

List of Bio-Plausible Gradient approxs: https://openreview.net/pdf?id=HJgPEXtIUS

Learning to Learn with Feedback and Local Plasticity
https://openreview.net/pdf?id=HklfNQFL8H

Structured and Deep Similarity Matching via Structured and Deep Hebbian Networks
http://papers.nips.cc/paper/9674-structured-and-deep-similarity-matching-via-structured-and-deep-hebbian-networks.pdf

Assessing the scalability of biologically-motivated deep learning algorithms
and architectures
https://papers.nips.cc/paper/8148-assessing-the-scalability-of-biologically-motivated-deep-learning-algorithms-and-architectures.pdf

### Neuroscience:

A mesoscale connectome of the mouse brain
https://www.nature.com/articles/nature13186

Neocortical layer 6, a review
https://www.frontiersin.org/articles/10.3389/fnana.2010.00013/full

### Vision:

**Yeah, this is what we are using undoubtably**
Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs
https://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns
(code at: https://github.com/dicarlolab/cornet)

http://www.brain-score.org/
(We definitely want some temporal abstraction e.g. recurrence, we also definitely want skip connections)

**DEFINITELY read this one:**
How well do deep neural networks trained on object recognition characterize the mouse visual system?
https://openreview.net/pdf?id=rkxcXmtUUS

**and this one**
Performance-optimized hierarchical models predict neural responses in higher visual cortex
https://www.pnas.org/content/111/23/8619

**And this... I particularly like this one...**
Neural Map: Structured Memory for Deep Reinforcement Learning
https://openreview.net/pdf?id=Bk9zbyZCZ

**Also this, we couldn't use this, but it has the right idea...**
Cognitive Mapping and Planning for Visual Navigation
http://openaccess.thecvf.com/content_cvpr_2017/papers/Gupta_Cognitive_Mapping_and_CVPR_2017_paper.pdf

Significance of feedforward architectural differences between the ventral visual stream and DenseNet
https://openreview.net/pdf?id=SkegNmFUIS

How well do deep neural networks trained on object recognition characterize the mouse visual system?
(Hint: They don't)
https://openreview.net/pdf?id=rkxcXmtUUS

Neural networks grown and self-organized by noise
http://papers.nips.cc/paper/by-source-2019-1100

Densely connected convolutional networks
https://arxiv.org/pdf/1608.06993.pdf

Surround Modulation: A Bio-inspired Connectivity Structure for Convolutional Neural Networks
http://papers.nips.cc/paper/9719-surround-modulation-a-bio-inspired-connectivity-structure-for-convolutional-neural-networks.pdf

A neural network model of flexible grasp movement generation
https://www.biorxiv.org/content/10.1101/742189v1.full.pdf

Deep Neural Networks and Visual Processing in the Rat 
https://www.researchgate.net/publication/326547016


### Motor Cortex

BioLSTMs
https://papers.nips.cc/paper/6631-cortical-microcircuits-as-gated-recurrent-neural-networks.pdf


### Hierarchical Methods:
My Idea:
Skip-Connection modulating pre-trained hierarchical model

Hierarchical Visuomotor Control of Humanoids
https://arxiv.org/pdf/1811.09656v1.pdf

**Deep Neuroethology of a Virtual Rodent**
https://arxiv.org/pdf/1911.09451.pdf

Hierarchical RL Using an Ensemble of Proprioceptive Periodic Policies
(These guys solve a "maze" using a humanoid)
https://openreview.net/pdf?id=SJz1x20cFQ

Learning Multi-level Hierarchies with Hindsight
https://arxiv.org/pdf/1712.00948.pdf

Sub-Policy Adaptation for Hierarchical Reinforcement Learning
https://openreview.net/pdf?id=ByeWogStDS

### RL
Off-Policy Actor-Critic with Shared Experience Replay
https://arxiv.org/pdf/1909.11583.pdf


## Things to keep in mind:
Neuron densities of mouse brain
https://www.frontiersin.org/articles/10.3389/fnana.2018.00083/full

Modulating lower level policies more than just A_t, but the latent space


To read:

http://papers.nips.cc/paper/8327-experience-replay-for-continual-learning.pdf


**Lateral**
https://www.pnas.org/content/114/32/8637



