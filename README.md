To use:
Install DGEB library / dependencies
Install rensa library: 

Then:

`python dgeb_retrieval_min_hash.py --k-mers 7 --num-perm 512`

The results will be printed to console.

Note: DGEB will complain about the "model" not having a "hf_name" attribute.  This is because MinHash is not a neural network, and is not on HuggingFace.