# copied and modified from https://github.com/microsoft/TRELLIS.2/blob/main/setup.sh

# Read Arguments
TEMP=`getopt -o h --long help,all,new-env,basic,flash-attn,nvdiffrast,cumesh,flexgemm,o-voxel,xformers -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
ALL=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
NVDIFFRAST=false
CUMESH=false
FLEXGEMM=false
OVOXEL=false
XFORMERS=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --all) ALL=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --all                   Run every step below (full inference environment)"
    echo "  --new-env               Create the 'lato2' conda env (python 3.10 + torch 2.6.0)"
    echo "  --basic                 Install core runtime deps (spconv, torch-scatter, open3d, ...)"
    echo "  --flash-attn            Install flash-attention (default attention backend)"
    echo "  --nvdiffrast            Build nvdiffrast   (required at o_voxel import time)"
    echo "  --cumesh                Build CuMesh       (required at o_voxel import time)"
    echo "  --flexgemm              Build FlexGEMM     (required at o_voxel import time)"
    echo "  --o-voxel               Build/install o_voxel (sparse-cloned from the TRELLIS.2 repo)"
    echo "  --xformers              (optional) Install xformers (alt attn backend / DINOv2 speedup)"
    return
fi

if [ "$ALL" = true ] ; then
    NEW_ENV=true
    BASIC=true
    FLASHATTN=true
    NVDIFFRAST=true
    CUMESH=true
    FLEXGEMM=true
    OVOXEL=true
fi

WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    return 1
fi

CONDA_EXE_PATH=""
for c in "$CONDA_EXE" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda ; do
    if [ -n "$c" ] && [ -x "$c" ] ; then CONDA_EXE_PATH="$c" ; break ; fi
done
if [ -z "$CONDA_EXE_PATH" ] ; then
    _c=$(command -v conda 2>/dev/null)
    if [ -x "$_c" ] ; then CONDA_EXE_PATH="$_c" ; fi
fi
if [ -n "$CONDA_EXE_PATH" ] ; then
    CONDA_BASE_DIR=$("$CONDA_EXE_PATH" info --base 2>/dev/null)
    if [ -n "$CONDA_BASE_DIR" ] && [ -f "$CONDA_BASE_DIR/etc/profile.d/conda.sh" ] ; then
        source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
    fi
elif [ "$NEW_ENV" = true ] ; then
    echo "Error: conda not found; cannot create a new env"
    return 1
fi

ensure_cuda() {
    if command -v nvcc > /dev/null 2>&1 ; then return 0 ; fi
    for cu in "$CUDA_HOME" /usr/local/cuda /usr/local/cuda-12.4 ; do
        if [ -n "$cu" ] && [ -x "$cu/bin/nvcc" ] ; then
            export CUDA_HOME="$cu"
            export PATH="$cu/bin:$PATH"
            return 0
        fi
    done
    return 1
}

ENV_NAME="${LATO_ENV:-lato2}"

if [ "$NEW_ENV" = true ] ; then
    conda create -n "$ENV_NAME" python=3.10 -y || { echo "Error: 'conda create' failed (a corrupted conda package cache is a common cause; try 'conda clean --all')"; return 1; }
    conda activate "$ENV_NAME" || { echo "Error: 'conda activate $ENV_NAME' failed"; return 1; }
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    elif [ "$PLATFORM" = "hip" ] ; then
        pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
fi

if [ "$BASIC" = true ] || [ "$FLASHATTN" = true ] || [ "$NVDIFFRAST" = true ] || [ "$CUMESH" = true ] || [ "$FLEXGEMM" = true ] || [ "$OVOXEL" = true ] || [ "$XFORMERS" = true ] ; then
    if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ] ; then
        conda activate "$ENV_NAME" || { echo "Error: could not activate env '$ENV_NAME'. Create it first with --new-env."; return 1; }
    fi
fi

if [ "$BASIC" = true ] ; then
    pip install numpy trimesh tqdm pillow ninja psutil opencv-python-headless huggingface_hub open3d==0.19.0 plyfile zstandard easydict
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install spconv-cu124==2.3.8
        pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[BASIC] spconv/torch-scatter have no prebuilt ROCm wheels here; install manually."
    fi
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        ensure_cuda || echo "[FLASHATTN] nvcc not found; ok if a prebuilt wheel is used, required for a source build."
        pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        ensure_cuda || echo "[NVDIFFRAST] nvcc not found; required to build."
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/nvdiffrast
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install /tmp/extensions/nvdiffrast --no-build-isolation --no-cache-dir
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$CUMESH" = true ] ; then
    ensure_cuda || echo "[CUMESH] nvcc not found; required to build."
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/CuMesh
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation --no-cache-dir
fi

if [ "$FLEXGEMM" = true ] ; then
    ensure_cuda || echo "[FLEXGEMM] nvcc not found; required to build."
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/FlexGEMM
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    pip install /tmp/extensions/FlexGEMM --no-build-isolation --no-cache-dir
fi

if [ "$OVOXEL" = true ] ; then
    mkdir -p /tmp/extensions
    ensure_cuda || echo "[O_VOXEL] nvcc not found; set CUDA_HOME to a CUDA toolkit and retry."
    rm -rf /tmp/extensions/TRELLIS.2
    git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/TRELLIS.2.git /tmp/extensions/TRELLIS.2
    git -C /tmp/extensions/TRELLIS.2 sparse-checkout set o-voxel
    git -C /tmp/extensions/TRELLIS.2 submodule update --init --recursive --depth 1
    pip install /tmp/extensions/TRELLIS.2/o-voxel --no-build-isolation --no-cache-dir
fi

if [ "$XFORMERS" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi
