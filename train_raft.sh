#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_raft.py --name raft-usdata \
                                            --restore_ckpt models/raft-things.pth \
                                            --stage usdata \
                                            --validation usdata \
                                            --num_steps 30000 \
                                            --batch_size 4 \
                                            --lr 0.0001 \
                                            --image_size 376 464 \
                                            --wdecay 0.00001

