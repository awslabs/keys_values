# What to do if Nvidia broken on P4 Instance after Reboot

## Fastest fix: boot into the old kernel where the driver already works

The dkms status output shows nvidia/570.133.20 is already installed for 6.8.0-1029-aws. You just need to boot into it:

```bash
# List available kernels
grep -E "menuentry|submenu" /boot/grub/grub.cfg | head -20

# Set the old kernel as default (adjust the index as needed)
sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-1029-aws"
sudo reboot
```

Or, to make it permanent until you fix the driver:
```bash
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-1029-aws"/' /etc/default/grub
sudo update-grub
sudo reboot
```

After rebooting, nvidia-smi should work immediately since that kernel already has the modules built.

## Proper fix (after you're back on a working kernel): upgrade the NVIDIA driver

NVIDIA 575+ supports kernel 6.17. Once you're back on 6.8.0:

```bash
# Check what's available
sudo apt-cache search nvidia-driver | grep -E "^nvidia-driver"

# Install a newer driver (e.g., 575 or 580)
sudo apt-get install -y nvidia-driver-575
sudo reboot
```

Then you can boot into 6.17 again and dkms autoinstall should succeed.

TL;DR: Reboot into 6.8.0-1029-aws first (it already has working drivers), then upgrade the NVIDIA driver before switching back to 6.17.
