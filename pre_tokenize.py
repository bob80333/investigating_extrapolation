import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

MAX_CHUNK = 128  # increase if you have lots of RAM and cpus

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data-directory", type=str, default="ao3_small_dataset/")
    parser.add_argument("--input-data-extension", type=str, default=".md")
    parser.add_argument("--output-data-extension", type=str, default=".tok")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("byte_tokenized_8k.json")

    files = list(Path(args.input_data_directory).rglob("*"+args.input_data_extension))

    files_chunked = []

    num_chunks = (len(files) // MAX_CHUNK) + 1

    for i in range(num_chunks):
        files_chunked.append(files[i*MAX_CHUNK:(i+1)*MAX_CHUNK])

    for chunk in files_chunked:
        print("Chunk of", len(chunk))
        if len(chunk) == 0:
            break
        texts = []
        print("Loading chunk of files")
        for file in tqdm(chunk):
            with open(file, 'r', encoding='utf-8') as reader:
                texts.append("<SOS>" + reader.read() + "<EOS>")

        print("Tokenizing chunk of files")
        start = time.time()
        tokenized = tokenizer.encode_batch(texts)
        end = time.time()
        print("Took", end-start, "to tokenize")

        print("Processing...")
        del texts  # save memory by deleting texts
        # save memory by replacing token object list with np arrays of ids
        tokenized = [np.array(tokens.ids) for tokens in tokenized]

        print("Saving chunk of files")
        for file, tokens in tqdm(zip(chunk, tokenized)):
            file = str(file.absolute()).replace(args.input_data_extension, args.output_data_extension)

            with open(file, 'wb') as writer:
                pickle.dump(tokens, writer)

    print("Complete")
