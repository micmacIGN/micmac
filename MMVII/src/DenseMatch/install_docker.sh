!/usr/bin/env bash


MODELPATH=models/finetune_PSMnet.tar


#clone PSMNet
git clone https://github.com/erupnik/PSMNet.git
cd PSMNet

#create new branch 'test', resetted to initial commit
git checkout -b test
git reset --hard 9ba1e36903f3ba2c99e5be8f03d31d2751a2cb33

#download a trained model
wget https://drive.google.com/uc?id=1JzVwoUuCdXfKmB26rPyV3vISRqgOUZxj -O ${MODELPATH}
cd ..

#add modules to the env
pip3 install -r requirements.txt
