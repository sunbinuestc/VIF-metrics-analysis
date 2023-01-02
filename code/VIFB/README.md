## How to use
### How to run algorithms
1. Please set the algorithms you want to run in util\configMethods.m
2. Please set the images you want to fuse in util\configImgsVI.m and util/configImgsIR.m, and change the path of these images
3. DLF and ResNet methods need MatConvNet to run. One should set the path to MatConvNet in run_ DLF.m (line 28) nad run_ ResNet.m (line 10), respectively.
4. DLF requires 'imagenet-vgg-verydeep-19.mat' to run. Please download it and put it inside methods\DLF
5. ResNet requires 'imagenet-resnet-50-dag.mat' to run. Please download it and put it inside methods\ResNet\models
6. To run GFF, please set your own path in run_GFF.m (line 17) 
7. To run MST_SR for the first time, please run "make" in the path "...\methods\MST_SR\sparsefusion\ksvdbox\ompbox\private". Similarly, to run RP_SR for the first time, run "make" in the path "...\methods\RP_SR\sparsefusion\ksvdbox\ompbox\private". To run NSCT_SR for the first time, run "make" in the path "...\methods\NSCT_SR\sparsefusion\ksvdbox\ompbox\private".
8. main_running.m is used to run the fusion algorithms. Please change the output path in main_running.m.
9. Enjoy!


### How to compute evaluation metrics
1. Please set the metrics you want to compute in util\configMetrics.m
2. compute_metrics.m is used to compute evaluation metrics. Please change the output path in compute_metrics.m
3. Enjoy!

### How to add algorithms (or fused images)
1. For methods written in MATLAB, please put them in the folder methods. For example, for method "ADF", put the codes inside a folder called "ADF", and put the folder "ADF" inside "Methods". Then change the main file of ADF to run_ADF.m. In run_ADF.m, please change the interface as according to examples provided in VIFB.
2. For algorithms written in Python or other languages, we suggest the users change the name of the fused images  according to examples provided and put them in the output folder. Then add the methods in util\configMethods.m. Then, the evaluation metrics can be computed.
