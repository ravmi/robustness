



metrics:
- accuracy (mean, ...)

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
