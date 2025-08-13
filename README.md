# Noise2Void Denoising

A package for denoising STEM experiments using the Noise2Void technique. The package will have the functionality to
train and apply trained Noise2Void models with several backbones/underlying architectures. It will also optimised.

As new experimental datasets are produced, a new module should be added to the `datasets` directory, which should
abstract over the file naming and directory structure for that particular experimental dataset.