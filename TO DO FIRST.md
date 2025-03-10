get detector-base
```bash
wget https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt
```

install requirements
```bash
# (on the top-level directory of this repository)
pip install -r requirements.txt
````

train
```bash
python -m detector.train
```

run the app
```bash
python -m detector.server detector-base.pt
```
