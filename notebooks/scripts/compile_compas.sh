#!/bin/bash

#wget https://github.com/TeamCOMPAS/COMPAS/archive/refs/heads/dev.zip
#unzip dev.zip
cd COMPAS-src/
make clean && make -f Makefile
