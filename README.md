<h1 align="center">Teamwork</h1>
<p align="center">
<a href="https://bin.samsartor.com/teamwork.pdf">📃 Paper</a> | <a href="https://samsartor.com/teamwork">🌐 Website</a> | <a href="https://bin.samsartor.com/teamwork_supplemental.pdf">🖼️ Supplemental</a> | <a href="https://huggingface.co/samsartor/teamwork-release">🤗 Models</a>
</p>
<p align="center"><img width="700px" src="https://samsartor.com/teamwork_teaser.svg" /></p>

The official implementation of "Teamwork: Collaborative Diffusion with Low-rank Coordination and Adaptation".

You can use this whole repository as a package with `uv add git+https://github.com/samsartor/teamwork[diffusers]`.
Alternatively, feel free to copy `minimal.py` as a starting-point for your own implementation.

```python
from teamwork.pipelines import TeamworkPipeline
from PIL import Image

pipe = TeamworkPipeline.from_checkpoint(
    'samsartor/teamwork-release',
    'decomposition_heterogeneous_sd3.safetensors',
).to('cuda')

generated = pipe({'image': Image.open('./demo/red_glass_sphere.png'})
```

See the notebooks folder for complete examples.

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
(LoRA) to jointly address both adaptation and coordination between the
different teammates. Furthermore Teamwork supports dynamic (de)activation
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
