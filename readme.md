<div align="center">

<h1>LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow</h1>

<p>
  <a href="https://lohhhha.github.io">Hang Long</a><sup>1,2,*</sup> &nbsp;
  <a href="https://tianhaozhao668.github.io">Tianhao Zhao</a><sup>1,2,*</sup> &nbsp;
  <a href="https://maymhappy.github.io">Junkai Lin</a><sup>1,2</sup> &nbsp;
  <a href="https://youjiazhang.github.io">Youjia Zhang</a><sup>1,2</sup> &nbsp;
  <a href="https://github.com/Ghuipeng">Huipeng Guo</a><sup>1</sup><br>
  <a href="https://github.com/PENGUINLIONG">Rendong Liang</a><sup>2</sup> &nbsp;
  <a href="https://bluestyle97.github.io">Jiale Xu</a><sup>2</sup> &nbsp;
  Jozef Hladký<sup>3</sup> &nbsp;
  <a href="https://niessnerlab.org">Matthias Nießner</a><sup>4</sup> &nbsp;
  <a href="https://yuanming.taichi.graphics">Yuanming Hu</a><sup>2</sup> &nbsp;
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

<div align="center">

[![🏠 Project Page](https://img.shields.io/badge/Project-Page-blue)](https://lohhhha.github.io/LATO.2/)
[![📄 Paper](https://img.shields.io/badge/Paper-arXiv-green)](https://arxiv.org/abs/2607.10623)
[![🤗 Model](https://img.shields.io/badge/Model-Hugging%20Face-yellow)](https://huggingface.co/0x4c48/LATO.2)

</div>

</div>

![LATO.2 teaser](./assets/teaser.png)

**LATO.2** factorizes mesh generation into a **vertex flow (V-Flow)** that generates vertex positions under a controllable vertex count, and a **topology flow (T-Flow)** that predicts connectivity from the realized vertices. This decomposition unlocks:

- **High-quality generation** (bottom left).
- **Part-wise generation** (bottom middle).
- **Topology-adaptive editing** (bottom right).

## Installation

1. Clone the repo:

    ```bash
    git clone https://github.com/LoHhhha/LATO.2.git
    cd LATO.2
    ```

2. Install the dependencies:

    The setup script creates a conda env named `lato2` (Python 3.10, PyTorch 2.6.0, CUDA 12.4). A CUDA toolkit (`nvcc`) is required to build the CUDA extensions.

    ```bash
    . ./setup.sh --all
    ```

## Checkpoints

Download the pretrained weights from the [Hugging Face Hub](https://huggingface.co/0x4c48/LATO.2) into `ckpt/`:

```bash
python scripts/ckpt_download.py
```

## Inference

All inference scripts take a directory of input meshes (`--mesh_dir`) and write results to `--out_dir`. Three bundled examples live in [`assets/example_mesh/`](./assets/example_mesh).

### Quick start

```bash
python scripts/e2e_inference.py \
  --mesh_dir assets/example_mesh \
  --out_dir outputs/e2e_run/example \
  --vert_num 2000
```

### End-to-end (`e2e_inference.py`)

Full V-Flow → T-Flow pipeline: input mesh → conditioning view → generated vertices → generated topology.
Outputs per mesh: `<id>_pred.obj` (generated mesh), `<id>_pred.ply` (vertices), the voxel-coordinate variants (`*_coords.*`, in `[0, 1024)`), and `<id>_render.png` (the DINOv2 conditioning view).

```bash
python scripts/e2e_inference.py \
  --mesh_dir <dir> \
  --out_dir outputs/e2e_run/<dir> \
  --vert_num <vert_num> \
  [--batch_size 4]
```

### V-Flow (`vflow_inference.py`)

First stage only — generates vertices from the conditioning view (optionally also reconstructs them via the VAE with `--reconstruct`). Outputs the generated / GT / reconstructed vertex PLYs plus the render.

```bash
python scripts/vflow_inference.py \
  --mesh_dir <dir> \
  --out_dir outputs/vflow_run/<dir> \
  --vert_num <vert_num> \
  [--batch_size 4]
```

### V-VAE (`vvae_inference.py`)

Vertex-VAE reconstruction only — encodes the input vertices and decodes them back at sub-voxel precision. Outputs the GT and reconstructed vertex PLYs.

```bash
python scripts/vvae_inference.py \
  --mesh_dir <dir> \
  --out_dir outputs/vvae_run/<dir> \
  [--batch_size 4]
```

### T-Flow (`tflow_inference.py`)

Second stage only — generates connectivity for the input vertices, conditioned on the voxel scaffold. Outputs `<id>_pred.obj` (generated mesh), the known vertices fed to the flow, and the conditioning field.

```bash
python scripts/tflow_inference.py \
  --mesh_dir <dir> \
  --out_dir outputs/tflow_run/<dir> \
  [--batch_size 4] \
  [--no-use_cond]
```

### Controlling the vertex count

The V-Flow stage (`e2e_inference.py` / `vflow_inference.py`) generates a controllable number of vertices:

- **Fixed target** (default): `--vert_num <N>` (default `2000`).
- **Scaled from the input**: `--use_gt_vert_count --scaler <s>` uses the input mesh's own vertex count multiplied by `<s>`.

Either way the count is clamped to `[--min_verts, --max_verts]` (defaults `200`–`5000`).

### Notes

> [!WARNING]
> Generated meshes may still contain holes and incorrect connectivity. We believe these artifacts can be largely resolved by scaling up.

- **Hardware**: runs in ~8 GB VRAM; end-to-end generation takes ~5 s per mesh on a single H800.
- **Image-to-structure**: to obtain the coarse voxel scaffold from a single image (instead of an existing mesh), you can directly reuse [TRELLIS](https://github.com/Microsoft/TRELLIS)'s sparse-structure stage.

## Acknowledgements

Our work builds upon these excellent repositories:

- [TripoSF](https://github.com/VAST-AI-Research/TripoSF)
- [TRELLIS](https://github.com/Microsoft/TRELLIS)
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [LATO](https://github.com/TianhaoZhao668/LATO)

## Citation

```bibtex
@misc{long2026lato2factorized3dmesh,
      title={LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow}, 
      author={Hang Long and Tianhao Zhao and Junkai Lin and Youjia Zhang and Huipeng Guo and Rendong Liang and Jiale Xu and Jozef Hladký and Matthias Nießner and Yuanming Hu and Wei Yang},
      year={2026},
      eprint={2607.10623},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2607.10623}, 
}
```
