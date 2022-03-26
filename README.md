



metrics:
- accuracy (mean, ...)
- class imbalance calculates the percentage of images in an epoch such that all pixels where assigned to one class (usually background)

robust

tests/

possible issues:
```
   'Could not import amp.'
```
try
```
pip uninstal apex
```
and then
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```
More here: https://stackoverflow.com/questions/66610378/unencryptedcookiesessionfactoryconfig-error-when-importing-apex


What is logged - Scalars:
- aug/acc - different kinds of accuracies on augmented set, averaged over entire augmented set
- aug/as_false - how many pixels were predicted as false and how many pixels should be predicted as false
- aug/loss_per_epoch - loss on augmented set per epoch
- train/loss_per_epoch -training loss averaged over batches per epoch
- train/loss_per_step - training loss per batch
- val/acc - different kinds of accuracies on validation set (no lighting), averaged over the entire validatoin set
- val/as_false - how many pixels were predicted as false and how many pixels should be predicted as false
- val/as_true - how many pixels were predicted as true and how many pixels should be predicted as true
- val/loss_per_epoch - loss per epoch on validation set (no light)

What is logged - debug samples:
- val/img/img - random image that we perform the test on
- val/guessed/img - white pixels are the pixels that the model predicted as candidates to pick up here, random sample on validation set
- val/truth/img - ground truth for the ranadom image
- aug/img/img
- aug/guessed/img
- aug/truth/img
