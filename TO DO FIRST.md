### First Step
get detector-base
```bash
wget https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt
```

install requirements
```bash
# (on the top-level directory of this repository)
pip install -r requirements.txt
```

train
```bash
python -m detector.train
```

### Running

run the app
```bash
python -m detector.server detector-base.pt
```

### Troubleshooting

If you encounter the error `OSError: [Errno 48] Address already in use`. 
Kill the process using the port with the following command:

```bash
lsof -i :8080
kill -9 <PID>
```
