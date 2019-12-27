#!/usr/bin/env bash
cd checkpoint
wget -c https://storage.googleapis.com/stylemixer/iter_80000.pth.tar
wget -c https://storage.googleapis.com/stylemixer/vgg_normalised.pth
cd .. 