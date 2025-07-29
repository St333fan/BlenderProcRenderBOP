# BlenderProcRenderBOP
Data and Script for Rendering BOP or other for ZeroShop

# Install
https://github.com/DLR-RM/BlenderProc/tree/main

# Local Install
```
git clone https://github.com/DLR-RM/BlenderProc
cd BlenderProc
pip install -e .

# Blender Install
sudo snap install blender --classic
find ~/.local/bin -name blenderproc
export PATH=$PATH:~/.local/bin

# Test install
blenderproc quickstart

# only first start
echo 'export PATH=/home/st3fan/blender/blender-4.2.1-linux-x64/custom-python-packages/bin:$PATH' >> ~/.bashrc
blenderproc quickstart # test install
```

# Go to BlenderProc folder and pull this repo
Place under a ./models folder your .ply files for example, the YCB-V fine_models:
https://drive.google.com/file/d/1l0BsaZgmtuRH_xIXE-0lO9j3T6P59g9u/view?usp=sharing



```
# Run rendering
blenderproc run render.py
```

