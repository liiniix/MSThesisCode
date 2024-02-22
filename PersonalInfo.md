# GPU CAIL - HPC 005 (Titan XP) 

    - Anydesk id : 1273940894
    - Password: CSEDU123456

      - Ubuntu username: HPC-005
    - password: CSEDU123456


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

    - CUDA 12.1 doesn't work
    - CUDA 11.8 works
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

    - Pytorch Geometric with CUDA 11.8
`conda install pyg -c pyg`

    - Anaconda environment
`conda create --name ThakiPytorchGeometric`

`conda activate ThakiPytorchGeometric`














# Huge problem faced during set-up on Masud Kamal vai's PC

PyTorch default version changed from 2.1 to 2.2

But Geo is on 2.1

Install Pytorch 2.1(with cuda 11.8, not 12 bc it is experimental) from previous version page.

then install matplotlib

then install wandb
`pip install wandb`
api key: `f9e59bb17a93f3143af93e7c4b8a4bc67e237d32`

also python 3.8






Microsoft Windows [Version 10.0.18362.30]
(c) 2019 Microsoft Corporation. All rights reserved.

C:\Users\Masud Karim>nvidia-smi
Fri Feb  2 11:45:56 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 536.23                 Driver Version: 536.23       CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080      WDDM  | 00000000:01:00.0  On |                  N/A |
|  0%   49C    P8              12W / 200W |    654MiB /  8192MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2352    C+G   ....Cortana_cw5n1h2txyewy\SearchUI.exe    N/A      |
|    0   N/A  N/A      5432    C+G   ...siveControlPanel\SystemSettings.exe    N/A      |
|    0   N/A  N/A      5588    C+G   ....Experiences.TextInput.InputApp.exe    N/A      |
|    0   N/A  N/A      5788    C+G   C:\Windows\explorer.exe                   N/A      |
|    0   N/A  N/A      7108    C+G   ...2txyewy\StartMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A      7244    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
|    0   N/A  N/A      9292    C+G   ...1.0_x64__8wekyb3d8bbwe\Video.UI.exe    N/A      |
|    0   N/A  N/A      9560    C+G   ..._8wekyb3d8bbwe\Microsoft.Photos.exe    N/A      |
+---------------------------------------------------------------------------------------+

C:\Users\Masud Karim>






`conda install -c conda-forge optuna`

`conda install pytorch-sparse -c pyg`




## For windows
`conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 pyg -c pytorch -c nvidia -c pyg`
`pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html`