  403  cd train
  404  mkdir test
  405  mkdri validation
  406  mkdir validation
  407  conda activate difusco
  408  python -u main.py gendata random None mis_train/data_er/train --model er --min_n 10 --max_n 30 --num_graphs 1000 --er_p 0.15
  409  mkdir -p /tmp/gpus
  410  touch /tmp/gpus/.lock
  411  touch /tmp/gpus/0.gpu
  412  python -u main.py solve kamis mis_train/data_er/train mis_train/data_er/train_annotations --time_limit 6000
  413  clear
  414  cd train
  415  mkdir test
  416  mkdir validation
  417  htop
  418  sudo apt install htop
  419  htop
  420  cond create --name datathon
  421  conda create --name datathon
  422  conda activate datathon
  423  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  424  conda install anaconda::seaborn
  425  conda install anaconda::pandas
  426  conda install anaconda::jupyter
  427  conda install conda-forge::jupyter
  428  jupyter notebook
  429  conda install anaconda::colorama
  430  jupyter notebook
  431  conda install tqdm
  432  jupyter notebook
  433  python /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
  434  python train/train_navigation.py 
  435  code .
  436  cd Documents/Cohet\ Files\ -\ Jahir\,\ Deeparghya/
  437  cd Deeparghya-pc/
  438  conda env list
  439  cd HetGPPO_ELIGN_Augment/
  440  conda env create -f environment.yml 
  441  export PIP_DEFAULT_TIMEOUT=100
  442  conda env create -f environment.yml 
  443  conda env list
  444  conda env remove hetgppo
  445  conda remove -n hetgppo --all
  446  conda env create -f environment.yml 
  447  conda activate hetgppo
  448  ls
  449  python train/train_navigation.py 
  450  code .
  451  nvcc --version
  452  cuda --version
  453  nvcc --version
  454  python train/train_navigation.py 
  455  clear
  456  python train/train_navigation.py 
  457  python train/train_sampling.py 
  458  clear
  459  python train/train_sampling.py
  460  clear
  461  python train/train_sampling.py
  462  clear
  463  python train/train_sampling.py
  464  clear
  465  python train/train_sampling.py
  466  clear
  467  python train/train_sampling.py
  468  clear
  469  python train/train_sampling.py
  470  clear
  471  python train/train_sampling.py
  472  clear
  473  python train/train_sampling.py
  474  clear
  475  python train/train_sampling.py
  476  clear
  477  python train/train_sampling.py
  478  clear
  479  python train/train_sampling.py
  480  clear
  481  python train/train_sampling.py
  482  clear
  483  python train/train_sampling.py
  484  clear
  485  ls
  486  cd train
  487  python train_reverse_transport.py 
  488  clear
  489  python train_reverse_transport.py 
  490  clear
  491  python train_reverse_transport.py 
  492  clear
  493  python train_reverse_transport.py 
  494  ls
  495  python train/train_reverse_transport.py 
  496  clear
  497  python train/train_reverse_transport.py 
  498  clear
  499  python train/train_reverse_transport.py 
  500  clear
  501  python train/train_sampling.py 
  502  clear
  503  python train/train_sampling.py 
  504  clear
  505  python train/train_sampling.py 
  506  clear
  507  nvidia-smi
  508  locate cuda
  509  sudo apt install plocate
  510  locate cuda
  511  whereis cuda
  512  locate cuda
  513  locate cuda/bin
  514  lspci | grep -i nvidia
  515  nvidia-smi
  516  nvcc --version
  517  sudo apt install nvidia-cuda-toolkit
  518  nvcc --version
  519  watch -n 2 ls
  520  watch -n 2 nvidia-smi 
  521  touch how_to_resolve_dependencies.txt
  522  getcwd
  523  pwd
  524  python3 -m venv local_lib
  525  source local_lib/bin/activate
  526  python --version
  527  python3.8 -m venv local_lib
  528  python38 -m venv local_lib
  529  deactivate
  530  sudo apt install python3.8
  531  clear
  532  sudo add-apt-repository ppa:deadsnakes/ppa
  533  sudo apt update
  534  sudo apt install python3.8
  535  python3.8 -m venv local_lib
  536  python3.8 -m pip install --upgrade setuptools wheel
  537  python --version
  538  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  539  python3.8 get-pip.py
  540  clear
  541  sudo add-apt-repository ppa:deadsnakes/ppa
  542  sudo apt update
  543  apt list --upgradable
  544  sudo apt install python3.8
  545  sudo apt install -u python3.8
  546  python3.8 --version
  547  python3.8 -m venv local_lib
  548  python3 -m venv local_lib
  549  source local_lib/bin/activate
  550  pip3 install -r requirements.txt
  551  pip install --upgrade pip
  552  deactivate
  553  clear
  554  locate python3.8
  555  wget https://www.python.org/ftp/python/3.8.15/Python-3.8.15.tgz
  556  tar -xzvf Python-3.8.15.tgz
  557  cd Python-3.8.15
  558  ./configure --enable-optimizations --prefix=/usr/local
  559  make -j8
  560  sudo make altinstall
  561  sudo add-apt-repository ppa:deadsnakes/ppa
  562  sudo apt update
  563  sudo apt install python3.9
  564  python3.9 --version
  565  clear
  566  python 3.9 -m venv local_lib
  567  python3.9 -m venv local_lib
  568  python3 --version
  569  python3.9 -m pip --version
  570  python3.9 get-pip.py
  571  clear
  572  sudo apt-get purge python3.9
  573  sudo apt-get autoremove
  574  sudo apt update
  575  sudo apt install python3.9
  576  python3.9 --version
  577  python3.9 -m pip --version
  578  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  579  clear
  580  python3.9 get-pip.py
  581  rm ~/local/share/trash/files/Python-3.2.8.15
  582  rm ~/local/share/Trash/files/Python-3.2.8.15
  583  rm ~/local/share/Trash/files/Python-3.8.15
  584  rm ~/.local/share/Trash/files/Python-3.8.15
  585  rm ~/.local/share/Trash/files/Python-3.2.8.15
  586  rm -r ~/.local/share/Trash/files/Python-3.2.8.15
  587  clear
  588  mv ~/.local/share/Trash/files/Python-3.8.15 .
  589  mv ~/.local/share/Trash/files/Python-3.8.15 /home/cail001usr/Desktop/Rizvee/pythons
  590  cd /home/cail001usr/Desktop/Rizvee/pythons/Python-3.8.15
  591  ./configure --enable-optimizations --prefix=/usr/local
  592  make -j8
  593  clear
  594  ./configure --enable-optimizations --prefix=/usr/local
  595  make -j8
  596  git pull
  597  git status
  598  git add *.gitignore
  599  git commit -m "removed tracking pyc files"
  600  git push
  601  sudo apt-get purge python3.8
  602  sudo apt-get autoremove
  603  sudo apt update
  604  clear
  605  sudo apt install python3.8
  606  python3.8 --version
  607  python --version
  608  pyt3hon --version
  609  python3 --version
  610  locate python 3.8
  611  clear
  612  dpkg -l | grep python3.8
  613  which python3.8
  614  python3.8 -m pip --version
  615  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  616  python3.8 get-pip.py
  617  sudo apt install zlib1g-dev
  618  python3.8 get-pip.py
  619  ls
  620  cd train
  621  python train_simple_
  622  ls
  623  python train_simple_spread.py 
  624  clear
  625  ls
  626  python train_simple_spread.py 
  627  python train_reverse_transport.py 
  628  clear
  629  python train_reverse_transport.py 
  630  clear
  631  python train_sampling.py 
  632  clear
  633  ls
  634  python train/train_navigation.py 
  635  sudo apt install zlib1g-dev
  636  python3.8 -c "import zlib; print(zlib)"
  637  clear
  638  wget https://www.python.org/ftp/python/3.8.15/Python-3.8.15.tgz
  639  tar -xf Python-3.8.15.tgz
  640  sudo apt update
  641  sudo apt install build-essential zlib1g-dev
  642  cd Python-3.8.15
  643  ./configure --with-zlib
  644  clear
  645  make -j8
  646  sudo make altinstall
  647  python3.8 --version
  648  python3.8 -m ensurepip
  649  python3.8 -m pip --version
  650  python3.8 -m venv  local_lib
  651  cd ..
  652  python3.8 -m venv  local_lib
  653  source local_lib/scripts/activate
  654  source local_lib/bin/activate
  655  pip install -r requirements.txt
  656  pip3 install -r requirements.txt
  657  sudo apt install libssl-dev
  658  deactivate
  659  cd Python-3.8.15/
  660  ./configure --with-ssl
  661  make
  662  sudo make install
  663  python3.8 -c "import ssl; print(ssl.OPENSSL_VERSION)"
  664  python3.8 venv local_lib
  665  python3.8 -m               venv local_lib
  666  cd ..
  667  python3.8 -m               venv local_lib
  668  source local_lib/bin/activate
  669  pip install -r requirements.txt
  670  clear
  671  pip install -r requirements.txt
  672  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install --upgrade pip
  673  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install setuptools
  674  pip install -r requirements.txt
  675  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  676  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch
  677  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  678  clear
  679  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  680  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch
  681  ./local_lib/bin/pip install torch-cluster
  682  ./local_lib/bin/pip install torch
  683  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch
  684  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip list
  685  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch=1.11.0
  686  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch==1.11.0
  687  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster==1.6.0
  688  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  689  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install --upgrade pip setuptools
  690  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  691  echo $PYTHONPATH
  692  deactivate
  693  conda deactivate
  694  python3.8 -m venv local_lib
  695  source local-lib/scripts/activate
  696  source local_lib/scripts/activate
  697  source local_lib/bin/activate
  698  pip install -r requirements.txt
  699  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install setuptools
  700  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install --upgrade pip
  701  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install setuptools
  702  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  703  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch
  704  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch-cluster
  705  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch torch-cluster
  706  pip install wheel
  707  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch torch-cluster
  708  pip install setuptools
  709  local_lib/bin/pip install -r setuptools
  710  local_lib/bin/pip installsetuptools
  711  local_lib/bin/pip install setuptools
  712  /home/cail001usr/Desktop/Rizvee/learning-diffusco/madrl/local_lib/bin/python3.8 -m pip install torch torch-cluster
  713  pip install torch-geometric==2.0.4
  714  torch-scatter==2.0.9
  715  torch-sparse==0.6.13
  716  torch-spline-conv==1.2.1
  717  torchaudio==0.11.0
  718  torchvision==0.12.0
  719  tornado==6.3.2
  720  tqdm==4.65.0
  721  traitlets==5.9.0
  722  typer==0.9.0
  723  typing-extensions==4.5.0
  724  tzdata==2023.3
  725  urllib3==2.0.2
  726  wcwidth==0.2.6
  727  zipp==3.15.0
  728  deactivate
  729  clear
  730  sudo apt-get update
  731  sudo apt-get install zlib1g-dev
  732  sudo apt-get install libssl-dev
  733  cd Python-3.8.15
  734  ./configure --with-ssl --with-zlib --with-ensurepip
  735  make
  736  ./configure --with-ssl --with-zlib --with-ensurepip --enable-optimizations
  737  make
  738  python3.8 -c "import ssl; print(ssl.OPENSSL_VERSION)"
  739  python3.8 -c "import zlib; print(zlib.ZLIB_VERSION)"
  740  python3.8 -m venv local_lib
  741  cd ..
  742  python3.8 -m venv local_lib
  743  source/bin/activate 
  744  clear
  745  local_lib/bin/activate
  746  source local_lib/bin/activate
  747  pip install -r requirements.txt 
  748  python3.8 -m pip install --upgrade setuptools
  749  sudo apt-get install build-essential
  750  deactivate
  751  python3.8 -m pip install --upgrade setuptools
  752  sudo apt-get install build-essential
  753  /usr/locaaaaall/bin/python3.8 -m pip install --upgrade pip
  754  /usr/local/bin/python3.8 -m pip install --upgrade pip
  755  python3.8          -m venv local_lllib
  756  python3.8          -m venv local_lib
  757  source local_lib/bin/activate
  758  pip install -r requirements.txt
  759  clear
  760  python train/train_navigation.py 
  761  clear
  762  python train/train_navigation.py 
  763  clear
  764  python train/train_navigation.py 
  765  clear
  766  python train/train_navigation.py 
  767  clear
  768  python train/train_navigation.py 
  769  clear
  770  python train/train_navigation.py 
  771  clear
  772  python train/train_navigation.py 
  773  clear
  774  python train/train_navigation.py 
  775  clear
  776  python train/train_navigation.py 
  777  clear
  778  python train/train_navigation.py 
  779  clear
  780  python train/train_navigation.py 
  781  clear
  782  python train/train_navigation.py 
  783  clear
  784  python train/train_flocking.py
  785  python train/train_flocking.py 
  786  clear
  787  python train/train_flocking.py 
  788  clear
  789  python train/train_flocking.py 
  790  clear
  791  python train/train_navigation.py 
  792  clear
  793  python train/train_navigation.py 
  794  nvidia-smi --id=1
  795  nvtop
  796  top
  797  conda create -n ThakiPytorchGeometric
  798  conda activate ThakiPytorchGeometric
  799  conda install python=3.8
  800  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  801  conda install pyg==2.4.0 -c pyg
  802  conda install torch-scatter==2.1.2 torch-sparse==0.6.18 -c pyg
  803  conda install pytorch-scatter==2.1.2 pytorch-sparse==0.6.18 -c pyg
  804  conda install matplotlib
  805  conda install wandb
  806  conda install wandb --channel conda-forge
  807  git checkout 
  808  fa0ab3f
  809  git checkout fa0ab3f
  810  python thesis_code_controller.py 
  811  git stash
  812  git checkout
  813  git checkout main
  814  python frontend.py 
  815  cd MakeEdgelist/CppHelper/
  816  g++ abul.cpp 
  817  ./a.out 
  818  cd ../..
  819  python frontend.py 
  820  cd MakeEdgelist/CppHelper/
  821  g++ abul.cpp 
  822  ./a.out 
  823  cd ../..
  824  python frontend.py 
  825  cd MakeEdgelist/CppHelper/
  826  g++ abul.cpp 
  827  ./a.out 
  828  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
  829  python train/train_flocking.py 
  830  clear
  831  python train/train_flocking.py 
  832  python train/train_flocking.py 
  833  clear
  834  python train/train_flocking.py 
  835  python train/train_joint_passage.py 
  836  clear
  837  python train/train_sampling.py 
  838  clear
  839  python
  840  clear
  841  python train/train_sampling.py 
  842  clear
  843  python train/train_sampling.py 
  844  clear
  845  python train/train_sampling.py 
  846  clear
  847  python train/train_sampling.py 
  848  clear
  849  python train/train_sampling.py 
  850  clear
  851  python train/train_sampling.py 
  852  clear
  853  python train/train_sampling.py 
  854  clear
  855  python train/train_sampling.py 
  856  clear
  857  python train/train_sampling.py 
  858  clear
  859  python train/train_sampling.py 
  860  clear
  861  python train/train_sampling.py 
  862  clear
  863  python train/train_sampling.py 
  864  clear
  865  python train/train_sampling.py 
  866  cler
  867  python train/train_sampling.py 
  868  clear
  869  python train/train_sampling.py 
  870  clear
  871  python train/train_sampling.py 
  872  clear
  873  python train/train_sampling.py 
  874  clear
  875  python train/train_sampling.py 
  876  clear
  877  python train/train_sampling.py 
  878  clear
  879  python train/train_sampling.py 
  880  clear
  881  python train/train_sampling.py 
  882  clear
  883  python train/train_sampling.py 
  884  clear
  885  python train/train_sampling.py 
  886  clear
  887  python train/train_sampling.py 
  888  clear
  889  python train/train_sampling.py 
  890  python train/train_flocking.py 
  891  clear
  892  python train/train_flocking.py 
  893  clear
  894  python 
  895  clear
  896  python train/train_flocking.py 
  897  clear
  898  python train/train_joint_passage.py 
  899  clear
  900  python train/train_joint_passage.py 
  901  clear
  902  python train/train_sampling.py 
  903  clear
  904  python train/train_sampling.py 
  905  clear
  906  python train/train_sampling.py 
  907  clear
  908  python train/train_sampling.py 
  909  clear
  910  python train/train_sampling.py 
  911  python train/train_reverse_transport.py 
  912  clear
  913  python train/train_reverse_transport.py 
  914  clear
  915  python train/train_reverse_transport.py 
  916  clear
  917  python train/train_joint_passage.py 
  918  python train/train_flocking.py 
  919  clear
  920  python train/train_flocking.py 
  921  clear
  922  python train/train_flocking.py 
  923  clear
  924  python train/train_joint_passage.py 
  925  clear
  926  python train/train_sampling.py 
  927  clear
  928  python train/train_sampling.py 
  929  clear
  930  python train/train_sampling.py 
  931  clear
  932  python train/train_sampling.py 
  933  clear
  934  python train/train_sampling.py 
  935  clear
  936  python train/train_sampling.py 
  937  clear
  938  python train/train_sampling.py 
  939  clear
  940  python train/train_sampling.py 
  941  clear
  942  python train/train_sampling.py 
  943  clear
  944  python train/train_sampling.py 
  945  python train/train_reverse_transport.py 
  946  clear
  947  python train/train_reverse_transport.py 
  948  clear
  949  python train/train_joint_passage.py 
  950  python train/train_flocking.py 
  951  clear
  952  python train/train_flocking.py 
  953  clear
  954  python train/train_joint_passage.py 
  955  clear
  956  python train/train_joint_passage.py 
  957  clear
  958  python train/train_navigation.py 
  959  clear
  960  python train/train_sampling.py 
  961  clear
  962  python train/train_sampling.py 
  963  clear
  964  python train/train_sampling.py 
  965  clear
  966  python train/train_sampling.py 
  967  clear
  968  python train/train_sampling.py 
  969  clear
  970  python train/train_sampling.py 
  971  clear
  972  python train/train_sampling.py 
  973  clear
  974  python train/train_sampling.py 
  975  clear
  976  python train/train_sampling.py 
  977  clear
  978  python train/train_sampling.py 
  979  python train/train_joint_passage.py 
  980  clear
  981  python train/train_joint_passage.py 
  982  clear
  983  python train/train_sampling.py 
  984  clear
  985  python train/train_sampling.py 
  986  clear
  987  python train/train_sampling.py 
  988  clear
  989  python train/train_sampling.py 
  990  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
  991  viewport
  992  vp
  993  reset
  994  hj
  995  japan
  996  reset
  997  python frontend.py
  998  cd MakeEdgelist/
  999  ls
 1000  cd CppHelper/
 1001  ls
 1002  g++ abul.cpp 
 1003  ./a.out 
 1004  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
 1005  g++ abul.cpp 
 1006  ./a.out 
 1007  sudo apt update
 1008  sudo apt update 
 1009  sudo apt upgrade 
 1010  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
 1011  for i in
 1012  ls LRGB/train
 1013  ls LRGB/train/*.txt
 1014  rm LRGB/train/*.txt
 1015  ls LRGB/train/*.txt
 1016  ls LRGB/train/
 1017  g++ abul.cpp 
 1018  ./a.out 
 1019  ls LRGB/train/*.txt
 1020  rm LRGB/train/*.txt
 1021  ls LRGB/train/*.txt
 1022  g++ abul.cpp 
 1023  ./a.out 
 1024  fd
 1025  g++ abul.cpp 
 1026  ./a.out 
 1027  g++ abul.cpp 
 1028  ./a.out 
 1029  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.4.1/python_files/deactivate/bash/envVars.txt
 1030  python frontend.py 
 1031  nvidia-smi 
 1032  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/deactivate/bash/envVars.txt
 1033  python frontend.py 
 1034  cd Documents/MSThesisCode/MakeEdgelist/CppHelper/LRGB/train/
 1035  ls
 1036  ls *.pt
 1037  ls *.ph
 1038  ls *.p
 1039  ls *.p*
 1040  ls *.pth
 1041  rm *.pth
 1042  ls *.pth
 1043  cd Documents/MSThesisCode/MakeEdgelist/CppHelper/LRGB/train/
 1044  ls *pth
 1045  rm *pth
 1046  git status
 1047  git checkout
 1048  git add .
 1049  git status
 1050  git add .
 1051  git status
 1052  git add
 1053  git add .
 1054  git status
 1055  git add .
 1056  git commit -m "Firstttt LRGB"
 1057  git config user.email "thaki240198cgc@gmail.com"
 1058  git config user.name "liiniix"
 1059  git commit -m "Firstttt LRGB"
 1060  git push origin main
 1061  python frontend.py 
 1062  ls
 1063  python frontend.py 
 1064  reset
 1065  python frontend.py 
 1066  reset
 1067  git  status
 1068  git add .
 1069  git commit -m "Before training on LRGB"
 1070  git push origin main
 1071  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/deactivate/bash/envVars.txt
 1072  python frontend.py 
 1073  pip install torcheval
 1074  python frontend.py 
 1075  python frontend.py 
 1076  rm test/*pth
 1077  rm train/*pth
 1078  rm val/*pth
 1079  rm test/*.edgelist
 1080  rm train/*.edgelist
 1081  rm val/*.edgelist
 1082  cd Documents/
 1083  cd MSThesisCode/MakeEdgelist/CppHelper/LRGB/
 1084  ls train/*.pth
 1085  ls test/*.pth
 1086  ls val/*.pth
 1087  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/deactivate/bash/envVars.txt
 1088  ls train/
 1089  python frontend.py 
 1090  cd MakeEdgelist/CppHelper/
 1091  g++ abul.cpp 
 1092  ./a.out 
 1093  g++ abul.cpp 
 1094  ./a.out 
 1095  g++ abul.cpp 
 1096  ./a.out 
 1097  ls
 1098  cd MakeEdgelist/CppHelper/
 1099  ls
 1100  cd LRGB_Pep/
 1101  ls
 1102  cd test/
 1103  ls
 1104  rm *.txt
 1105  ls
 1106  python frontend.py 
 1107  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.8.1/python_files/deactivate/bash/envVars.txt
 1108  \
 1109  git status
 1110  git history
 1111  git log
 1112  git reset --hard  1a14ea
 1113  git status
 1114  git reset --hard cbf8cee
 1115  git checkout  1a14ea
 1116  git reset --hard cbf8cee
 1117  it reset HEAD~ 
 1118  git reset HEAD~ 
 1119  git status
 1120  git add .
 1121  git commit -m "QquotttttttttttttaFirst2"
 1122  git push       origin main
 1123  ls
 1124  tree
 1125  git rm -r --cached MakeEdgelist/CppHelper/LRGB_Pep/
 1126  git rm -r --cached MakeEdgelist/CppHelper/LRGB_Pep
 1127  ls
 1128  git log
 1129  git push origin main 
 1130  git status
 1131  git log -one-line
 1132  git log --one-line
 1133  git log --oneline 
 1134  git checkout -b main1 c570105
 1135  git status
 1136  git push origin main1
 1137  conda install pandas
 1138  conda install seaborn
 1139  conda install ogb
 1140  git log
 1141  git branch ogbTest ad703
 1142  git checkout ogbtest
 1143  git checkout ogbTest
 1144  git status
 1145  git add .gitignore lrgb_test.ipynb 
 1146  git status
 1147  git stash
 1148  git status
 1149  git stash pop
 1150  git status
 1151  git add .gitignore lrgb_test.ipynb 
 1152  git status
 1153  git commit -m "Before going to ogbTest branch"
 1154  git push origin main1
 1155  git status 
 1156  git stash
 1157  git checkout ogbTest
 1158  git stash pop
 1159  git status
 1160  git stash list
 1161  git log
 1162  git revert ad703
 1163  git merge --abort
 1164  git reset --hard ad703
 1165  git status
 1166  reset
 1167  git status
 1168  git add ogb_meg_test.ipynb 
 1169  /home/cail001usr/anaconda3/envs/ThakiPytorchGeometric/bin/python /home/cail001usr/Documents/MSThesisCode/lrgb_test_normal.py
 1170  git status
 1171  git add .
 1172  git commit -m "QuotaFirst"
 1173  git checkout HEAD^ -- MakeEdgelist/CppHelper/LRGB_Pep
 1174  git checkout HEAD^ -- MakeEdgelist/CppHelper/LRGB_Pep/
 1175  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.10.0-linux-x64/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.10.0-linux-x64/python_files/deactivate/bash/envVars.txt
 1176  cd MakeEdgelist/CppHelper/
 1177  g++ abul.cpp 
 1178  ./a.out 
 1179  g++ abul.cpp 
 1180  ./a.out 
 1181  g++ abul.cpp 
 1182  ./a.out 
 1183  cd ../..
 1184  git status
 1185  cd MakeEdgelist/CppHelper/
 1186  g++ abul.cpp 
 1187  ./a.out 
 1188  g++ abul.cpp 
 1189  ./a.out 
 1190  g++ abul.cpp 
 1191  ./a.out 
 1192  g++ abul.cpp 
 1193  ./a.out 
 1194  g++ abul.cpp 
 1195  ./a.out 
 1196  g++ abul.cpp 
 1197  ./a.out 
 1198  g++ abul.cpp 
 1199  ./a.out 
 1200  g++ abul.cpp 
 1201  ./a.out 
 1202  g++ abul.cpp 
 1203  ./a.out 
 1204  g++ abul.cpp 
 1205  ./a.out 
 1206  git status
 1207  git commit -m "added ogbMeg dataset test"
 1208  git push origin ogbTest
 1209  git status
 1210  ls
 1211  g++ abul.cpp 
 1212  ./a.out 
 1213  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.12.3-linux-x64/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.12.3-linux-x64/python_files/deactivate/bash/envVars.txt
 1214  git status
 1215  reset
 1216  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.12.3-linux-x64/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.12.3-linux-x64/python_files/deactivate/bash/envVars.txt
 1217  head ogbmeg.edgelist 
 1218  tail ogbmeg.edgelist 
 1219  haed cora.edgelist
 1220  head cora.edgelist
 1221  reset
 1222  head ogbmeg.edgelist 
 1223  head ogbmeg.edgelist | sed 's/{[^}]*}//g'
 1224  head ogbmeg.edgelist | sed -r ' :L; s/({ "}]*)\"(([^"'}]*")*)([^"'}]*})/124/g; tL; '
 1225  head citeseer.edgelist
 1226  head ogbmeg.edgelist | sed 's/{[^}]*}/{}/g'
 1227  cat ogbmeg.edgelist | sed 's/{[^}]*}/{}/g'
 1228  cat ogbmeg.edgelist | sed -i 's/{[^}]*}/{}/g'
 1229  sed -i 's/{[^}]*}/{}/g' ogbmeg.edgelist 
 1230  head ogbmeg.edgelist 
 1231  ls
 1232  ls -l
 1233  sed -i 's/{[^}]*}/{}/g' acm.edgelist
 1234  head acm.edgelist
 1235  sed -i 's/{[^}]*}/{}/g' acm.edgelist
 1236  head acm.edgelist
 1237  python frontend.py 
 1238  cd
 1239  nvidia-smi
 1240  mkdir mahmud26
 1241  cd mahmud26
 1242  cd sum/
 1243  python -m venv .venv
 1244  source .venv/bin/activate
 1245  jupyter lab
 1246  pip3 install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 1247  jupyter lab
 1248  pip install git+https://github.com/csebuetnlp/normalizer
 1249  jupyter lab
 1250  nvtop
 1251  sudo apt install nvtop
 1252  nvtop
 1253  sudo apt install chrome-gnome-shell
 1254  nvtop
 1255  nvcc --version
 1256  nvtop
 1257  sudo apt install xrdp
 1258  systemctl status xrdp
 1259  sudo ufw status
 1260  sudo ufw allow 3389
 1261  sudo ufw status
 1262  sudo ufw status verbose 
 1263  sudo ufw allow 22
 1264  nvtop
 1265  source .venv/bin/ac
 1266  source .venv/bin/activate
 1267  pip install datasets
 1268  reboot
 1269  sudo apt update
 1270  sudo apt upgrade
 1271  sudo add-apt-repository universe
 1272  gsettings set org.gnome.shell.extensions.dash-to-dock extend-height false
 1273  gsettings set org.gnome.shell.extensions.dash-to-dock autohide true
 1274  gsettings set org.gnome.shell.extensions.dash-to-dock hide-delay 9999999
 1275  ls
 1276  cd Downloads/
 1277  ls
 1278  wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
 1279  sudo dpkg -i google-chrome-stable_current_amd64.deb
 1280  sudo apt-get install -f
 1281  sudo apt upgrade
 1282  sudo apt autoremove
 1283  cd 
 1284  cd Desktop/mahmud26/
 1285  ls
 1286  cd sum/
 1287  source .venv/bin/activate
 1288  jupyter lab
 1289  pip install --quiet datasets
 1290  jupyter lab
 1291  nvtop
 1292  ls
 1293  mkdir res 
 1294  cp results/* cp/
 1295  cp results/* cp
 1296  cp results/* res/
 1297  cp -rf results/* res/
 1298  nvtop
 1299  cd Desktop/mahmud26/sum/
 1300  source .venv/bin/activate
 1301  jupyter lab
 1302  cd Desktop/mahmud26/sum/
 1303  ls
 1304  source .venv/bin/ac
 1305  source .venv/bin/activate
 1306  jupyter lab
 1307  cd ../mt/
 1308  ls
 1309  python3 -m venv .venv
 1310  souce .venv/bin/activate
 1311  source .venv/bin/activate
 1312  jupyter lab
 1313  nvtop
 1314  logout
 1315  cd Desktop/mahmud26/sum/
 1316  ls
 1317  source .venv/bin/activate
 1318  jupyter lab
 1319  sudo apt-get update
 1320  sudo apt-get install -y cuda
 1321  sudo apt-get install cuda
 1322  sudo apt-get install -y cuda
 1323  sudo apt-get -y install cuda
 1324  cuda --version
 1325  sudo apt-get clean
 1326  sudo apt-get autoremove
 1327  sudo apt upgrade
 1328  sudo apt-get install nvidia-cuda-toolkit
 1329  cuda --version
 1330  sudo apt-get install -y libcudnn8
 1331  source .venv/bin/ac
 1332  source .venv/bin/activate
 1333  pip install tensorflow-gpu
 1334  python --version
 1335  pip3 install tensorflow-gpu
 1336  sudo apt-get install -y libcudnn8
 1337  sudo apt update && rm -rf /var/lib/apt/lists/*
 1338  sudo add-apt-repository multiverse
 1339  sudo apt update
 1340  sudo apt install nvidia-cuda-toolkit   
 1341  sudo apt upgrade
 1342  sudo apt autoremove
 1343  pip install --upgrade pip
 1344  python3 -m pip install tensorflow==2.13.*
 1345  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
 1346  apt-cache search --names-only '^nvidia-driver-[0-9]{3}$'
 1347  nvidia-smi
 1348  cd Desktop/mahmud26/
 1349  ls
 1350  ls mt/
 1351  git clone https://github.com/CAIL-DU/summarizer.git
 1352  ls
 1353  cp summarizer/t5/BanglaLongT5_corrected.ipynb mt/
 1354  ls
 1355  cd mt/
 1356  source .venv/bin/activate
 1357  jupyter lab
 1358  exigt
 1359  exit
 1360  nvtop
 1361  exit
 1362  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.12.3-linux-x64/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.12.3-linux-x64/python_files/deactivate/bash/envVars.txt
 1363  git status
 1364  git log
 1365  git status
 1366  git add .
 1367  git commit -m "tested for Heterogeneous graph, but failed"
 1368  git push origin ogbTest
 1369  conda deactivate
 1370  conda activate ThakiPytorchGeometric
 1371  conda list
 1372  conda env export > environment.yml
 1373  git status
 1374  git add .
 1375  git commit -m "added environment.yml"
 1376  git push origin ogbTest
 1377  cd Desktop/mahmud26/mt/
 1378  ls
 1379  source .venv/bin/activate
 1380  jupyter lab
 1381  sudo apt update
 1382  sudo apt upgrade
 1383  nvtop
 1384  sudo ufw status
 1385  sudo ufw allow 22
 1386  sudo
 1387  sudo -s
 1388  /bin/python3 /home/cail001usr/.vscode/extensions/ms-python.python-2024.14.1-linux-x64/python_files/printEnvVariablesToFile.py /home/cail001usr/.vscode/extensions/ms-python.python-2024.14.1-linux-x64/python_files/deactivate/bash/envVars.txt
 1389  sudo apt update 
 1390  sudo apt upgrade 
 1391  reset
 1392  RESET
 1393  reset
 1394  cd ~
 1395  cat .bashrc 
 1396  exit
 1397  ls
 1398  reset
 1399  parso
 1400  history
 1401  xrpd
 1402  history > a.sh
