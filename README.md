# KKY-MPV


1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create uv environment. Use Python 3.11.
```shell
uv sync
```

3. Download the dataset and change the structure:
```
FungiTastic
|- FungiTastic-FewShot (move folder from the /images folder)
|   |- train
|   |   |- fullsize
|   |- val
|   |   |- fullsize
|   |- test
|       |- fullsize
|- metadata
    |- FungiTastic-FewShot
        |- .csv files
```

4. Run script to create embeddings (Update paths / constants inside first. Must login to HF to download DINOv3).
```shell
bash ./scripts/schedule_ge.sh
```

5. Run ```02-Evaluate-embeddings-views.ipynb``` notebook to produce predictions / submissions.