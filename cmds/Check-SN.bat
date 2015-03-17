python ../convnet.py ^
--layer-def=../example-layers/layers.gc2.cfg ^
--layer-params=../example-layers/layer-params.gc2.cfg ^
--data-provider=dummy-cn-192 ^
--check-grads=1 ^
--gpu=1

@PAUSE