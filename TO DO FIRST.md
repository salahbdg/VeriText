```bash
wget https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt
```

```bash
# (on the top-level directory of this repository)
pip install -r requirements.txt
python -m detector.train
```

```bash
python -m detector.server detector-base.pt
````
