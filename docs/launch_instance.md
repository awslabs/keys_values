# Launch P4 Instance for Experimentation


## Current Instances and Volumes

### Instance `P4Research`

* Instance ID: `i-01be7aa867826a644`
* File system ID: `fs-0016c9964fcfc99f7`
* VPC: `vpc-0619b17e`
* Subnet ID: `subnet-65fffa1c`
* AZ: `us-west-2b`

```bash
ssh -i "matthis_deeplearning_uswest2.pem" ubuntu@ec2-35-85-224-176.us-west-2.compute.amazonaws.com
```

### Instance `P4Research2`

* Instance ID: `??`
* File system ID: `fs-0186b686e7dffc35b`
* VPC: `vpc-0619b17e`
* Subnet ID: `subnet-124f5848`
* AZ: `us-west-2c`

```bash
ssh -i "matthis_deeplearning_uswest2.pem" ubuntu@ec2-34-209-209-37.us-west-2.compute.amazonaws.com
```


## Launch and start instance

* Type: `p4d.24xlarge`
* AMI: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)`
* Volume size: 180 GiB
* Security group: `sg-0b00b6174eab4d62b (launch-wizard-16)`


## Create EFS volume

Usually, instances occupy different subnets and AZs, so cannot share the same
volume. It is easiest to create one volume per instance.

### Step 1: Create the EFS File System (AWS Console)

1. Go to **AWS Console → EFS → Create file system**
2. Click **Customize** for more options
3. Configure:
   - **Name**: give it a meaningful name
   - **Storage class**: Standard or One Zone (cheaper, single AZ)
   - **VPC**: select the **same VPC as your EC2 instance**
4. Click **Next**

### Step 2: Configure the Mount Target

1. In the **Network** section, select:
   - The **same Availability Zone** as your EC2 instance
   - The **same subnet** as your EC2 instance
   - A **Security Group** : `sg-01a150984c1045fbf (launch-wizard-11)`
2. Click **Next → Next → Create**

### Step 3: Configure Security Groups

**Note**: If the same SGs are used for instance and volume, this should
not be necessary to do more than once.

**On the EFS Security Group:**

Add an **inbound rule**:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| NFS | TCP | 2049 | EC2's Security Group ID (or EC2's private IP) |

**On the EC2 Security Group:**

Add an **outbound rule** (usually already open):

| Type | Protocol | Port | Destination |
|------|----------|------|-------------|
| NFS | TCP | 2049 | EFS Security Group ID (or `0.0.0.0/0`) |

###  General

* Security group of EFS: `sg-01a150984c1045fbf (launch-wizard-11)`
* Security group of EC2: `sg-0b00b6174eab4d62b (launch-wizard-16)`


## Mount EFS volume

Relevant docs:
* https://docs.aws.amazon.com/efs/latest/ug/installing-amazon-efs-utils.html
* https://github.com/aws/efs-utils?tab=readme-ov-file#on-other-linux-distributions

This is a bit harder than normal because we use an Ubuntu AMI.

### Step 1: Install the EFS Mount Helper on the EC2 Instance

SSH into your instance and run:

```bash
curl https://amazon-efs-utils.aws.com/efs-utils-installer.sh | sudo sh -s -- --install
```

Test:
```bash
mount.efs --version
```

Create a Mount Point
```bash
sudo mkdir -p /mnt/efs
```

### Step 2: Mount the EFS Volume

Replace `fs-0123456789abcdef0` with your file system ID.

Using EFS Mount Helper:
```bash
sudo mount -t efs -o tls fs-0123456789abcdef0:/ /mnt/efs
```

Verify the mount:
```bash
df -h | grep efs
# or
mount | grep efs
```

### Step 3: Make the Mount Persistent (Survives Reboots)

Edit `/etc/fstab`:
```bash
sudo nano /etc/fstab
```

Add this line at the bottom:
```
# Using EFS mount helper
fs-0123456789abcdef0:/ /mnt/efs efs _netdev,tls 0 0
```

> ⚠️ The `_netdev` option is **important** — it tells the system to wait for network before mounting.

Test the fstab entry without rebooting:
```bash
sudo mount -fav
```

Set Permissions (Optional)
```bash
# Give your user ownership
sudo chown -R ubuntu:ubuntu /mnt/efs
```

Setup directories:
```bash
cd /mnt/efs
mkdir -p out/finetune
cd out/finetune
mkdir data
mkdir neurips_exp
cd /home/ubuntu
mkdir -p out/finetune
cd out/finetune
ln -s /mnt/efs/out/finetune/data data
ln -s /mnt/efs/out/finetune/neurips_exp neurips_exp
```

## Clone Repository, Install Virtual Environment

This is copied from [README.md](./README.md), but modified in order to peg certain
versions of `torch, torchvision`. This is because the CUDA drivers installed on
the P4 instance do not work with `PyTorch 2.11` installed by default. See also
https://pytorch.org/.

```bash
cd /home/ubuntu
mkdir git
mkdir virtenvs
cd git
git clone https://github.com/awslabs/keys_values.git
cd ../virtenvs
python3 -m venv keys_values
. keys_values/bin/activate
pip install --upgrade pip
pip3 install torch==2.10.0 torchmetrics==1.8.2 torchvision==0.25.0
pip install 'litgpt[all,test,extra]'
cd ../git/keys_values
pip install -e .
```

Run tests:
```bash
pytest test/
```
