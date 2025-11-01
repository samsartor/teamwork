# Teamwork

The official implementation of "Teamwork: Collaborative Diffusion with Low-rank Coordination and Adaptation".

**Currently only a minimal implementation is provided. The full implementation will be
released before SIGGRAPH Asia 2025**

## Abstract

Large pretrained diffusion models can provide strong priors beneficial for
many graphics applications. However, generative applications such as neural
rendering and inverse methods such as SVBRDF estimation and intrinsic
image decomposition require additional input or output channels. Current
solutions for channel expansion are often application specific and these
solutions can be difficult to adapt to different diffusion models or new tasks.
This paper introduces Teamwork: a flexible and efficient unified solution for
jointly increasing the number of input and output channels as well as adapt-
ing a pretrained diffusion model to new tasks. Teamwork achieves channel
expansion without altering the pretrained diffusion model architecture by
coordinating and adapting multiple instances of the base diffusion model
(i.e., teammates). We employ a novel variation of Low Rank-Adaptation
(LoRA) to jointly address both adaptation and coordination between the dif-
ferent teammates. Furthermore Teamwork supports dynamic (de)activation
of teammates. We demonstrate the flexibility and efficiency of Teamwork
on a variety of generative and inverse graphics tasks such as inpainting,
single image SVBRDF estimation, intrinsic decomposition, neural shading,
and intrinsic image synthesis.

## Citation

```
@conference{Sartor:2025:TCD,
    author    = {Sartor, Sam and Peers, Pieter},
    title     = {Teamwork: Collaborative Diffusion with Low-rank Coordination and Adaptation},
    month     = {December},
    year      = {2025},
    booktitle = {ACM SIGGRAPH Asia Conference Proceedings},
}
```
