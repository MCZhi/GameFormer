# Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving

[Zhiyu Huang](https://mczhi.github.io/), [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Jingda Wu](https://wujingda.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 

[AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)

## Abstract

Predicting the future states of surrounding traffic participants and planning a safe, smooth, and socially compliant trajectory accordingly is crucial for autonomous vehicles. There are two major issues with the current autonomous driving system: the prediction module is often decoupled from the planning module and the cost function for planning is hard to specify and tune. To tackle these issues, we propose an end-to-end differentiable framework that integrates prediction and planning modules and is able to learn the cost function from data. Specifically, we employ a differentiable nonlinear optimizer as the motion planner, which takes as input the predicted trajectories of surrounding agents given by the neural network and optimizes the trajectory for the autonomous vehicle, thus enabling all operations in the framework to be differentiable including the cost function weights. The proposed framework is trained on a large-scale real-world driving dataset to imitate human driving trajectories in the entire driving scene and validated in both open-loop and closed-loop manners. The open-loop testing results reveal that the proposed method outperforms the baseline methods across a variety of metrics and delivers planning-centric prediction results, allowing the planning module to output close-to-human trajectories. In closed-loop testing, the proposed method shows the ability to handle complex urban driving scenarios and robustness against the distributional shift that imitation learning methods suffer from. In particular, we find that joint training of planning and prediction modules achieves better performance than planning with a separate trained prediction module in both open-loop and closed-loop tests. Moreover, the ablation study indicates that the learnable components in the framework are essential to ensure stability and planning performance. 

## Method Overview
The proposed framework consists of two parts. First, we build up a holistic neural network to embed the history states of agents and scene context into high-dimensional spaces, encode the interactions between agents and the scene context using Transformer modules, and finally decode different future predicted trajectories and their probabilities. Second, we employ a differentiable optimizer as a motion planner to explicitly plan a future trajectory for the AV according to the most-likely prediction result and initial motion plan. Since the motion planner is differentiable, the gradient from the planner can be backpropagated to the prediction module and the cost function weights can also be learned with the objective to imitate human driving trajectories.

<img src="./src/overview.png" style="width:90%;">

## Closed-loop testing

### Cruising

| <video muted controls width=380> <source src="./src/473df4d0702d0d61.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/ebf548112b4155bd.mp4"  type="video/mp4"> </video> |

### Traffic light

| <video muted controls width=380> <source src="./src/cf4c93a51f255da.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/5af7263eeae8cc38.mp4"  type="video/mp4"> </video> |

### Turning

| <video muted controls width=380> <source src="./src/cf966e6cb27802a3.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/4b7a175072d54d11.mp4"  type="video/mp4"> </video> |

### Emergency

| <video muted controls width=380> <source src="./src/3d43fef395cc20fd.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/8ea14bd2d120c0e4.mp4"  type="video/mp4"> </video> |

### Interaction

| <video muted controls width=380> <source src="./src/60a9ff3762d159d9.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/a7ec124caa56fbda.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/73993339ce528fd0.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/f135a85bfd8b8190.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/a1f64ce4c1a9f9f8.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/9ba3d538754e86bf.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./src/124ae6902ef59cea.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./src/84b40c7a7d2fb9b0.mp4"  type="video/mp4"> </video> |

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
