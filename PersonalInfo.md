# GPU CAIL - HPC 005 (Titan XP) 

    - Anydesk id : 1273940894
    - Password: CSEDU123456

    - Ubuntu username: HPC-005
    - password: CSEDU123456

```
(ThakiPytorchGeometric) cail005@HPC005:~/Documents/github/MSThesisCode$ nvidia-smi

Wed Jan 24 18:26:27 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.223.02   Driver Version: 470.223.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN Xp     Off  | 00000000:01:00.0  On |                  N/A |
| 23%   34C    P8    17W / 250W |    340MiB / 12194MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1982      G   /usr/lib/xorg/Xorg                143MiB |
|    0   N/A  N/A      2223      G   /usr/bin/gnome-shell               52MiB |
|    0   N/A  N/A      4125      G   ...RendererForSitePerProcess      142MiB |
+-----------------------------------------------------------------------------+
(ThakiPytorchGeometric) cail005@HPC005:~/Documents/github/MSThesisCode$ 
```

    - CUDA 12.1 doesn't work
    - CUDA 11.8 works
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

    - Pytorch Geometric with CUDA 11.8
`conda install pyg -c pyg`

    - Anaconda environment
`conda create --name ThakiPytorchGeometric`

`conda activate ThakiPytorchGeometric`