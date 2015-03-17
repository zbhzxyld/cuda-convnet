python ../convnet.py ^
--data-path=G:\\ftp\\cifar-10-py-colmajor ^
--data-path-test=G:\\ftp\\cifar-10-py-colmajor ^
--save-path=./tmp ^
--test-range=5 ^
--train-range=1-4 ^
--layer-def=G:\byang\cuda-convnet-L2\cuda-convnet\example-layers\layers-conv-local-13pct.cfg ^
--layer-params=G:\byang\cuda-convnet-L2\cuda-convnet\example-layers\layer-params-conv-local-13pct.cfg ^
--data-provider=cifar-cropped ^
--test-freq=13 ^
--crop-border=4 ^
--gpu=1 ^
--epochs=100



@PAUSE