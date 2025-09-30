# Noise2Void Denoising

A package for denoising STEM experiments using the Noise2Void technique. The package will have the functionality to
train and apply trained Noise2Void models with several backbones/underlying architectures. Training is
performed across several GPUs/nodes using PyTorch's `DistributedDataParallel`.

As new experimental datasets are produced, a new module should be added to the `datasets` directory, which should
abstract over the file naming and directory structure for that particular experimental dataset.

TODO: Move all prediction plotting to the `Dataset`'s methods, rather than handling in the prediction script.