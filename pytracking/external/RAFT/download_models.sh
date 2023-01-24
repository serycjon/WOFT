#!/bin/bash
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip

cp /home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/100000_raft-rob-100k.pth models
