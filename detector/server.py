import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
import subprocess
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import json
import fire
import torch
from urllib.parse import urlparse, unquote, parse_qs
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device: str = None

available_models = {
    "roberta": {
        "tokenizer": RobertaTokenizer.from_pretrained("roberta-base"),
        "model": RobertaForSequenceClassification.from_pretrained("roberta-base")
    },
    "bert": {
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "model": AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    },
    "distilbert": {
        "tokenizer": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
        "model": AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    },
    "xlnet": {
        "tokenizer": AutoTokenizer.from_pretrained("xlnet-base-cased"),
        "model": AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased")
    },
    "albert": {
        "tokenizer": AutoTokenizer.from_pretrained("albert-base-v2"),
        "model": AutoModelForSequenceClassification.from_pretrained("albert-base-v2")
    },
    "electra": {
        "tokenizer": AutoTokenizer.from_pretrained("google/electra-base-discriminator"),
        "model": AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator")
    }
}

def log(*args):
    print(f"[{os.environ.get('RANK', '')}]", *args, file=sys.stderr)

class RequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        # Parse the path and query
        path = urlparse(self.path).path
        query_params = parse_qs(urlparse(self.path).query)

        if path == '/':
            # Serve the HTML file for the root path
            self.begin_content('text/html')
            html = os.path.join(os.path.dirname(__file__), 'index.html')
            self.wfile.write(open(html).read().encode())
            return
        elif path == '/api/check':
            # Handle API request for checking text
            query = query_params.get('query', [''])[0]  # Get the 'query' parameter from URL
            model_name = query_params.get('model', ['roberta'])[0]  # Default to 'roberta'

            print(f"Received query: {query}, Model: {model_name}")
            if not query:
                self.send_error(400, "Query parameter 'query' is required")
                return

            if model_name not in available_models:
                self.send_error(400, f"Model '{model_name}' is not supported")
                return

            self.begin_content('application/json;charset=UTF-8')

            # Load the appropriate model
            selected_model = available_models[model_name]["model"]
            selected_tokenizer = available_models[model_name]["tokenizer"]

            tokens = selected_tokenizer.encode(query)
            all_tokens = len(tokens)
            tokens = tokens[:selected_tokenizer.model_max_length - 2]
            used_tokens = len(tokens)
            tokens_tensor = torch.tensor(
                [(selected_tokenizer.bos_token_id or selected_tokenizer.cls_token_id)] + tokens +
                [(selected_tokenizer.eos_token_id or selected_tokenizer.sep_token_id)]
            ).unsqueeze(0)
            mask = torch.ones_like(tokens_tensor)

            with torch.no_grad():
                outputs = selected_model(tokens_tensor.to(device), attention_mask=mask.to(device))
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
                probs = logits.softmax(dim=-1)
                fake, real = probs.detach().cpu().flatten().numpy().tolist()

                # Calculate perplexity and burstiness scores
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    perplexity_score = torch.exp(outputs.loss).item()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    labels = torch.tensor([1]).to(device)  # Assuming real label for perplexity calculation
                    loss = loss_fct(logits.view(-1, selected_model.config.num_labels), labels.view(-1))
                    perplexity_score = torch.exp(loss).item()

                burstiness_score = self.calculate_burstiness(probs)

                # Identify AI-written sentences
                ai_lines = self.identify_ai_sentences(query, selected_model, selected_tokenizer)
                print(f"AI lines: {ai_lines}")

            # Prepare and return JSON response
            response_data = dict(
                all_tokens=all_tokens,
                used_tokens=used_tokens,
                real_probability=real,
                fake_probability=fake,
                perplexity_score=perplexity_score,
                burstiness_score=burstiness_score,
                ai_lines=ai_lines
            )

            log('Response:', response_data)

            self.wfile.write(json.dumps(response_data).encode())
            return
        else:
            # Handle 404 for other paths
            self.send_error(404, "Not Found")

    def calculate_burstiness(self, probs):
        # Calculate burstiness as the variance in the probabilities of tokens
        variance = torch.var(probs, dim=-1).mean().item()
        return variance

    def identify_ai_sentences(self, text, selected_model, selected_tokenizer):
        sentences = text.split('\n')
        ai_lines = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():  # Skip empty lines
                continue

            tokens = selected_tokenizer.encode(sentence)
            tokens_tensor = torch.tensor(
                [(selected_tokenizer.bos_token_id or selected_tokenizer.cls_token_id)] + tokens +
                [(selected_tokenizer.eos_token_id or selected_tokenizer.sep_token_id)]
            ).unsqueeze(0)
            mask = torch.ones_like(tokens_tensor)

            with torch.no_grad():
                outputs = selected_model(tokens_tensor.to(device), attention_mask=mask.to(device))
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                sentence_probs = logits.softmax(dim=-1)
                fake, real = sentence_probs.detach().cpu().flatten().numpy().tolist()

                print(f"Sentence: {sentence}, Fake: {fake}, Real: {real}")
                if fake > real:
                    ai_lines.append(i)
        return ai_lines

    def begin_content(self, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def log_message(self, format, *args):
        log(format % args)

def serve_forever(server, device, checkpoint):
    log('Process has started; loading models...')
    for model_name, components in available_models.items():
        components["model"] = components["model"].to(device)
        data = torch.load(checkpoint, map_location='cpu')
        components["model"].load_state_dict(data['model_state_dict'], strict=False)
        components["model"].eval()
    globals()['device'] = device

    log(f'Ready to serve at http://localhost:{server.server_address[1]}')
    server.serve_forever()

def main(checkpoint, port=8080, device='cuda' if torch.cuda.is_available() else 'cpu'):
    if checkpoint.startswith('gs://'):
        print(f'Downloading {checkpoint}', file=sys.stderr)
        subprocess.check_output(['gsutil', 'cp', checkpoint, '.'])
        checkpoint = os.path.basename(checkpoint)
        assert os.path.isfile(checkpoint)

    print(f'Loading checkpoint from {checkpoint}')
    # data = torch.load(checkpoint, map_location='cpu')

    # model_name = 'roberta-large' if data['args']['large'] else 'roberta-base'
    # model = RobertaForSequenceClassification.from_pretrained(model_name)
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # model.load_state_dict(data['model_state_dict'], strict=False)
    # model.eval()

    print(f'Starting HTTP server on port {port}', file=sys.stderr)
    server = HTTPServer(('0.0.0.0', port), RequestHandler)

    # Avoid calling CUDA API before forking; doing so in a subprocess is fine.
    num_workers = int(subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.cuda.device_count())']))

    if num_workers <= 1:
        serve_forever(server, device, checkpoint)
    else:
        print(f'Launching {num_workers} worker processes...')

        subprocesses = []

        for i in range(num_workers):
            os.environ['RANK'] = f'{i}'
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{i}'
            process = Process(target=serve_forever, args=(server, device, checkpoint))
            process.start()
            subprocesses.append(process)

        del os.environ['RANK']
        del os.environ['CUDA_VISIBLE_DEVICES']

        for process in subprocesses:
            process.join()

if __name__ == '__main__':
    fire.Fire(main)
