# FungiCLEF 2025
Code for the winning solution to the [FungiCLEF 2025 competition](https://www.kaggle.com/competitions/fungi-clef-2025/overview).

## Conda env
This code utilizes a conda environment and Jupyter notebooks.


To recreate the env locally:
```bash
conda env create --name fungiclef2025 -file environment.yml
```

To add the env to your Jupyter kernels:
```bash
conda activate fungiclef2025
python -m ipykernel install --user --name=fungiclef2025
```

NOTE: You can name the env and the kernel whatever you'd like, but `fungiclef2025` is the kernel name in the notebooks if you don't want to have to fuss with switching kernels.

Notebooks:
- 01-create-embeddings-cache-beit-dinov2b.ipynb - create combined embeddings using the FungiTastic-BEiT and DINOv2-B models
- 01-create-embeddings-cache-dinov2l.ipynb - create embeddings using the DINOv2-L model
- 01-create-embeddings-cache-samh.ipynb - create embeddings using the SAM-ViT-H model
- 02-merge-cached-embeddings.ipynb - merge the embeddings from the first 3 notebooks and save the combined embeddings
- 03-make-submissions.ipynb - create submissions for the competition from the cached embeddings

These notebooks were run on a Linux desktop with an RTX 3090 and 128 GB of RAM. As an alternative to creating the embeddings in 3 passes, the embeddings can be created without batching and/or utilizing mixed precision, both of which will prevent out of memory errors. Inference at full precision without batching is fairly slow. Additionally, the combination of FungiTastic-BEiT and DINOv2-B will create embeddings that perform nearly as well as the combination of all 4 models and these two models are significantly leaner.

Simplified scripts: TBD

Paper and bibtex citation: TBD
