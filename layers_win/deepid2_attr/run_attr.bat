python ../convnet.py ^
--data-path=E:\database\webface\batches\part2\ ^
--data-path-E:\database\webface\batches\part2\ ^
--data-provider=xgw-deepid-attr-online ^
--layer-def=F:\Projects\deeplearning\cuda-convnet-L2\cuda-convnet\deepid2_attr\net-define-deepid2-attr-parallel.cfg ^
--layer-params=F:\Projects\deeplearning\cuda-convnet-L2\cuda-convnet\deepid2_attr\net-param-deepid2-attr-parallel.cfg ^
--save-path=EF:\Projects\deeplearning\cuda-convnet-L2\cuda-convnet\models\deepid2_attr\ ^
--test-range=901-930 ^
--train-range=1-900 ^
--test-freq=50 ^
--epochs=1


@PAUSE