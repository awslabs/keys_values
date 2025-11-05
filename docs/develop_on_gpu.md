# Notes how to develop and run code on GPU instances

## Setup

### Create EC2 instance

`DevelopmentWithGPU`:
* AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)
* Resource type: `g5.12xlarge` (4 GPUs, 24 GB; 192 GB RAM)

`DevelopmentWithGPUMoreMemory`:
* AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)
* Resource type: `g6e.16xlarge` (1 GPU, 48 GB; 512 GB RAM)

`P4Research`:
* AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)
* Resource type: `p4d.24xlarge` (8 GPU, 80 GB; 1152 GB RAM)

Don't forget to stop when not needed anymore!

NO! It may be better to just keep it running!

Next:

* Copy convenience files, such as `~/.gitconfig`
* Clone `litgpt` fork and `valkeyrie` repos
* Create venv with dependencies for these

GPU instances and memory per GPU, and RAM:

* `g5.12xlarge`:  24 GB (4 GPUs), 192 GB RAM
* `g5.16xlarge`:  24 GB (1 GPU),  256 GB RAM
* `g6e.12xlarge`: 48 GB (4 GPUs), 384 GB RAM
* `g6e.16xlarge`: 48 GB (1 GPU),  512 GB RAM
* `p4d.24xlarge`: 80 GB (8 GPUs: A100), 1152 GB RAM

Disk: 450 GiB, EBS

### Setup on instance

First make sure you ensure GitHub access on the instance. Next, we need some
modifications of `LitGPT`, which can obtained from a fork:
```bash
mkdir git
cd git
git clone git@github.com:mseeger/litgpt.git
cd litgpt
git checkout keys_values
cd ../..
mkdir virtenvs
cd virtenvs
python3 -m venv keys_values
. keys_values/bin/activate
pip install --upgrade pip
cd ../git/litgpt
pip install -e .[all,test,extra]
```

Next, you need to copy the source code from the package to the instance.
For whatever reasons, GitFarm packages cannot be cloned to an EC2 instance
other than a devbox. Devboxes, of course, do not run on GPU instances, which
are terribly hard to get anyway, and they run some age-old "Amazon Linux",
whereas you need the latest AMIs for serious deep learning work.

Maybe the easiest approach is to use remote development from an IDE, which
will copy the code to the instance automatically. See below for `PyCharm`.

Once you found a way to sync your source code to the instance regularly,
say to `git/Valkeyrie`, complete the virtual environment setup:
```bash
. virtens/keys_values/bin/activate
cd git/Valkeyrie
pip install -e .
```

### When running

This avoids GPU memory fragmentation:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Configure Deployment in PyCharm

https://www.jetbrains.com/help/pycharm/tutorial-deployment-in-product.html

This allows to develop code locally, which gets synced to the instance. This
seems to work fine.

Note:

* Have to change the SSH configuration of the deployment server every time
  an instance is started again. This also allows to switch between different
  instances, as long as the remote paths are all the same.
