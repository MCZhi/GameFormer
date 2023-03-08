# GameFormer

[Zhiyu Huang](https://mczhi.github.io/), [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 

[AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)

## Abstract

Autonomous vehicles operating in complex real-world environments require accurate predictions of interactive behaviors between traffic participants. While existing works focus on modeling agent interactions based on their past trajectories, their future interactions are often ignored. This paper addresses the interaction prediction problem by formulating it with hierarchical game theory and proposing the GameFormer framework to implement it. Specifically, we present a novel Transformer decoder structure that uses the prediction results from the previous level together with the common environment background to iteratively refine the interaction process. Moreover, we propose a learning process that regulates an agent's behavior at the current level to respond to other agents' behaviors from the last level. Through experiments on a large-scale real-world driving dataset, we demonstrate that our model can achieve state-of-the-art prediction accuracy on the interaction prediction task. We also validate the model's capability to jointly reason about the ego agent's motion plans and other agents' behaviors in both open-loop and closed-loop planning tests, outperforming a variety of baseline methods.

## Method Overview
The proposed framework draws inspiration from the hierarchical game-theoretic modeling of agent interactions. The framework encodes the historical states of agents and maps as background information. A level-0 agent's future trajectories are decoded independently, based on the initial modality query. At level-k, an agent responds to all other level-(k-1) agents. Scene context encoding is obtained via a Transformer-based encoder. The level-0 decoder takes the modality embedding and agent history encodings as query and outputs level-0 future trajectories and scores. The level-k decoder incorporates a self-attention module to model the level-(k-1) future interactions and appends this information to the scene context encoding.

<img src="./src/method overview.png">

## Interaction Prediction


## Closed-loop testing

| <video muted controls width=380> <source src="./src/473df4d0702d0d61.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/ebf548112b4155bd.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/cf4c93a51f255da.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/5af7263eeae8cc38.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/cf966e6cb27802a3.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/4b7a175072d54d11.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/cf966e6cb27802a3.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/4b7a175072d54d11.mp4"  type="video/mp4"> </video> |


## Citation
```
@article{huang2022differentiable,
  title={Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving},
  author={Huang, Zhiyu and Liu, Haochen and Wu, Jingda and Lv, Chen},
  journal={arXiv preprint arXiv:2207.10422},
  year={2022}
}
```

## Contact

If you have any questions, feel free to contact us (zhiyu001@e.ntu.edu.sg).
