python ../convnet.py ^
--data-path=J:\\byang\\data\\DeepID2+\\p2\\batches ^
--data-path-test=J:\\byang\\data\\DeepID2+\\p2\\batches ^
--data-provider=xgw-1-2x-4213 ^
--layer-def=G:/byang/cuda-convnet-L2/exps/L2-SN/net-define-Joint-m0-4213-deepid2+.cfg ^
--layer-params=G:/byang/cuda-convnet-L2/exps/L2-SN/net-params-Joint-0.0025-deepid2+.cfg ^
--save-path=I:/byang/data/DeepID2/official/p2/nets/deepid2+/ ^
--test-range=1-1 ^
--train-range=2-3 ^
--test-freq=2 ^
--max-filesize=10000 ^
--gpu=0 ^
--conserve-mem=1 ^
--epochs=1


@PAUSE