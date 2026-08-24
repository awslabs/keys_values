# Copy Files between EFS Volumes

## Initial prompt

I'd like to copy files from one AWS EFS volume to another. The volumes are in different
AZs with different subnets. Each volume is mounted on a separate EC2 instance.

Source instance and volume:

* Instance ID: `i-01dee00f42a643ea7`
* File system ID: `fs-0186b686e7dffc35b`
* VPC: `vpc-0619b17e`
* Subnet ID: `subnet-124f5848`
* AZ: `us-west-2c`

Target instance and volume:

* Instance ID: `i-01be7aa867826a644`
* File system ID: `fs-0016c9964fcfc99f7`
* VPC: `vpc-0619b17e`
* Subnet ID: `subnet-65fffa1c`
* AZ: `us-west-2b`

Security groups for instances and volumes are:

* Security group of EFS: `sg-01a150984c1045fbf (launch-wizard-11)`
* Security group of EC2: `sg-0b00b6174eab4d62b (launch-wizard-16)`

The files to be copied are in `/mnt/efs/helmet/longtrain` on the source volume.
They should be copied to `/mnt/efs/helmet/longtrain/copy` on the target volume.

## Answer:

Strategy
Both instances are in the same VPC (vpc-0619b17e), so they can communicate directly
over private IPs. The cleanest approach is rsync over SSH from the source instance
to the target instance.

### Step 1 — Allow SSH between the instances

The EC2 security group (sg-0b00b6174eab4d62b) must allow inbound SSH (port 22) from
the source instance. Add an inbound rule:

* Type: SSH (TCP 22)
* Source: <private IP of i-01dee00f42a643ea7>   # or the whole VPC CIDR, or the SG itself

* You can do this via the console or CLI:

```bash
# Get source instance's private IP
aws ec2 describe-instances --instance-ids i-01dee00f42a643ea7 \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text

# Add inbound rule (replace 10.x.x.x with the IP above)
aws ec2 authorize-security-group-ingress \
  --group-id sg-0b00b6174eab4d62b \
  --protocol tcp --port 22 \
  --cidr 10.x.x.x/32
```

### Step 2 — Get the target instance's private IP

```bash
aws ec2 describe-instances --instance-ids i-01be7aa867826a644 \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text
```

### Step 3 — Copy the SSH key to the source instance (if needed)

If the target requires a different key pair, copy that .pem to the source instance:

```bash
scp -i your-key.pem your-key.pem ubuntu@<source-private-ip>:~/.ssh/
chmod 600 ~/.ssh/your-key.pem   # run on source instance
```

### Step 4 — Run rsync from the source instance

SSH into the source instance, then:

```bash
rsync -avz \
  -e "ssh -i ~/.ssh/your-key.pem -o StrictHostKeyChecking=no" \
  /mnt/efs/helmet/longtrain/ \
  ubuntu@<target-private-ip>:/mnt/efs/helmet/longtrain/copy/
```

Key flags: -a preserves permissions/timestamps, -v verbose, -z compresses in transit.
The trailing / on the source means "copy contents of this directory" rather than
the directory itself.

### Alternative: mount both EFS volumes on one instance

If you'd rather avoid SSH key logistics, you can mount the target EFS directly on
the source instance:

```bash
# On source instance
sudo mkdir -p /mnt/efs_target
sudo mount -t efs -o tls fs-0016c9964fcfc99f7:/ /mnt/efs_target

# Then copy locally
rsync -av /mnt/efs/helmet/longtrain/ /mnt/efs_target/helmet/longtrain/copy/
This requires that the source instance's security group (sg-0b00b6174eab4d62b) is added as an allowed source in the EFS security group (sg-01a150984c1045fbf) on port 2049 (NFS). Since both instances already share the same SG sg-0b00b6174eab4d62b, and the EFS SG allows that SG as a source, this likely already works — worth checking the EFS inbound rules first.
```

The dual-mount approach is simpler if the network rules permit it; the
rsync-over-SSH approach works reliably regardless of EFS network rules.


## Question

I'd like to do the rsync-over-SSH approach.
I don't have any `.pem` files in `~/.ssh` on the source instance, but I have `~/.ssh/authorized_keys`
there. Can you update the instructions?

## Answer

Since your private key lives on your local machine (the source instance just has authorized_keys), the cleanest solution is SSH agent forwarding — your local SSH agent is forwarded through your session to the source instance, so it can authenticate to the target without any key file on disk.

Step 1 — SSH to the source instance with agent forwarding

ssh -A ubuntu@<source-private-ip>
The -A flag forwards your local SSH agent into the session.

Step 2 — Verify the agent is forwarded (optional sanity check)

ssh-add -l   # should list your key(s)
Step 3 — Ensure your public key is in the target's authorized_keys
The same public key you use to reach the source must also be in ~/.ssh/authorized_keys on the target instance. If it isn't, from your local machine:


ssh-copy-id ubuntu@<target-private-ip>
# or manually: cat ~/.ssh/id_rsa.pub | ssh ubuntu@<target-private-ip> "cat >> ~/.ssh/authorized_keys"
Step 4 — Run rsync from the source instance

rsync -avz \
  /mnt/efs/helmet/longtrain/ \
  ubuntu@<target-private-ip>:/mnt/efs/helmet/longtrain/copy/
No -i flag needed — rsync will use the forwarded agent automatically.

If agent forwarding isn't an option, the alternative is to generate a dedicated key on the source
instance:

```bash
# On source instance
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# Add that output to ~/.ssh/authorized_keys on the target instance, then rsync as above
```

## Question

I generated a dedicated key on the source instance as instructed, appended the output of
`cat ~/.ssh/id_ed25519.pub` to `~/.ssh/authorized_keys` on the target instance, then did this on
the source instance:

```bash
(keys_values) ubuntu@ip-172-31-0-209:~$ rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
  /mnt/efs/helmet/longtrain/ \
  ubuntu@172.31.26.205:/mnt/efs/helmet/longtrain/copy/
ubuntu@172.31.26.205: Permission denied (publickey).
rsync: connection unexpectedly closed (0 bytes received so far) [sender]
rsync error: unexplained error (code 255) at io.c(232) [sender=3.2.7]
```

What is wrong?

OK: Problem was `ec2-user` instead of `ubuntu`. Now, this works!

