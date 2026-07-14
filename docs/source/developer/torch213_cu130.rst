PyTorch 2.13 and CUDA 13.0 development environment
===================================================

This setup keeps the existing PyTorch 2.11/CUDA 12.8 checkout and virtual
environment unchanged.  The paths below are specific to the ``xixi``
workspace.

Isolated checkout and environment
---------------------------------

Create a worktree and virtual environment once::

    cd /home/user2/data/xixi/nunchaku
    git worktree add -b mxfp4-torch213-cu130 \
      /home/user2/data/xixi/nunchaku-torch213-cu130 mxfp4

    /home/user2/data/bin/uv venv \
      /home/user2/data/xixi/.venv-torch213-cu130 \
      --python 3.12

The original paths remain:

* source: ``/home/user2/data/xixi/nunchaku``
* environment: ``/home/user2/data/xixi/.venv``

Install PyTorch and the CUDA compiler
-------------------------------------

Install the cu130 PyTorch wheels and a CUDA 13.0 compiler inside the new
environment::

    export UV_CACHE_DIR=/home/user2/data/xixi/.cache/uv
    export VENV=/home/user2/data/xixi/.venv-torch213-cu130

    /home/user2/data/bin/uv pip install --python "$VENV/bin/python" \
      torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
      --index-url https://download.pytorch.org/whl/cu130

    /home/user2/data/bin/uv pip install --python "$VENV/bin/python" \
      'cuda-toolkit[nvcc,cccl]==13.0.3' ninja packaging setuptools wheel

The CUDA Python wheels currently contain ``libcudart.so.13`` but not the
unversioned development linker name.  Add it only inside this virtual
environment::

    export CUDA_ROOT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
    test -e "$CUDA_ROOT/lib/libcudart.so" || \
      ln -s libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"

Build Nunchaku
--------------

Initialize the worktree's submodules, select the environment-local CUDA
toolchain, and build the editable package::

    cd /home/user2/data/xixi/nunchaku-torch213-cu130
    git submodule update --init --recursive

    export VENV=/home/user2/data/xixi/.venv-torch213-cu130
    export CUDA_ROOT="$VENV/lib/python3.12/site-packages/nvidia/cu13"
    export CUDA_HOME="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
    export NUNCHAKU_INSTALL_MODE=FAST
    export MAX_JOBS=16

    /home/user2/data/bin/uv pip install --python "$VENV/bin/python" \
      -e . --no-build-isolation

``NUNCHAKU_INSTALL_MODE=FAST`` builds only for GPUs visible during the build.
On the RTX 5090 this produces ``sm_120a`` code.  Use ``ALL`` only when a wheel
must contain every architecture supported by Nunchaku; it takes longer and
uses more temporary disk space.

Activate and verify
-------------------

Each new shell must select the new environment and its CUDA libraries::

    source /home/user2/data/xixi/.venv-torch213-cu130/bin/activate
    export CUDA_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
    export CUDA_HOME="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

    python - <<'PY'
    from importlib.metadata import version

    import nunchaku
    import torch
    import torchvision

    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("torchvision:", torchvision.__version__)
    print("nunchaku:", version("nunchaku"))
    print("nunchaku path:", nunchaku.__file__)
    print("GPU:", torch.cuda.get_device_name())
    print("capability:", torch.cuda.get_device_capability())
    PY

Run the focused MXFP4 tests from the new worktree::

    cd /home/user2/data/xixi/nunchaku-torch213-cu130
    python -m pytest tests/test_mxfp4_precision.py -q

Do not run ``uv pip install`` without ``--python`` until the intended
environment has been activated.  Otherwise the package can be installed into
the old PyTorch 2.11 environment.
