#!/usr/bin/env bash
cd checkpoint
wget -c https://storage.googleapis.com/stylemixer/vgg_normalised.pth
mkdir styleMixer_bw1_style3.00_cont3.00_iden1.00_cx3.00_1
cd styleMixer_bw1_style3.00_cont3.00_iden1.00_cx3.00_1
wget -c https://storage.googleapis.com/stylemixer/iter_80000.pth.tar
cd ../..

