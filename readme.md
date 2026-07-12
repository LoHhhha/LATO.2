<div align="center">

<h1>LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow</h1>

<p>
  <a href="https://github.com/LoHhhha">Hang Long</a><sup>1,2,*</sup> &nbsp;
  <a href="https://tianhaozhao668.github.io">Tianhao Zhao</a><sup>1,2,*</sup> &nbsp;
  <a href="https://github.com/MayMhappy">Junkai Lin</a><sup>1,2</sup> &nbsp;
  <a href="https://youjiazhang.github.io">Youjia Zhang</a><sup>1,2</sup> &nbsp;
  Huipeng Guo<sup>1</sup><br>
  <a href="https://github.com/PENGUINLIONG">Rendong Liang</a><sup>2</sup> &nbsp;
  <a href="https://bluestyle97.github.io">Jiale Xu</a><sup>2</sup> &nbsp;
  Jozef Hladký<sup>3</sup> &nbsp;
  <a href="https://niessnerlab.org">Matthias Nießner</a><sup>4</sup> &nbsp;
  <a href="https://weiyang-hust.github.io">Wei Yang</a><sup>1,†</sup>
</p>

<p>
  <sup>1</sup>Huazhong University of Science and Technology &nbsp;·&nbsp;
  <sup>2</sup>Meshy AI &nbsp;·&nbsp;
  <sup>3</sup>Independent Researcher &nbsp;·&nbsp;
  <sup>4</sup>Technical University of Munich
</p>

<p><sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author</p>
<p><em>This work was completed during internships at Meshy AI.</em></p>

<p>
  <img src="https://img.shields.io/badge/Inference%20Code-Coming%20Soon-lightgrey" alt="Inference code coming soon">
  <img src="https://img.shields.io/badge/Pretrained%20Weights-Coming%20Soon-lightgrey" alt="Pretrained weights coming soon">
</p>

</div>

![LATO.2 teaser](./assets/teaser.png)

We present **LATO.2**, which factorizes mesh generation into a vertex flow (V-Flow) generating vertex positions under a controllable vertex count, and a topology flow (T-Flow) predicting connectivity from realized vertices. It supports high-quality generation (bottom left), part-wise generation at scalable resolution (bottom middle), and topology-adaptive editing (bottom right).

## Repository status

> [!IMPORTANT]
> **Inference code and pretrained checkpoints will be released soon.**

- [ ] Inference code
- [ ] Pretrained checkpoints

## Abstract

Flow matching over carefully designed latent representations has recently emerged as a powerful paradigm for topology-aware mesh generation. Existing approaches, however, model vertices and connectivity jointly in a joint latent space, entangling continuous vertex geometry with discrete combinatorial structure; this complicates flow learning and manifests as drifting vertices and broken surfaces. We present **LATO.2**, a factorized flow matching framework that decomposes mesh generation into a vertex flow followed by a connectivity flow conditioned on the realized vertices, with both stages anchored to a shared coarse voxel scaffold. Dedicated VAEs underpin the two stages, recovering vertices at sub-voxel precision and embedding discrete connectivity into a continuous latent space. We demonstrate two advantages unique to this factorization: (i) part-wise generation, in which the scaffold is partitioned and each part synthesized at full latent capacity, yielding substantially higher-resolution meshes than a monolithic latent permits; and (ii) topology-adaptive editing, in which manipulating first-stage vertices induces the corresponding connectivity without re-optimization. Experiments show that LATO.2 surpasses state-of-the-art topology-aware mesh generators in geometric fidelity and connectivity quality.
