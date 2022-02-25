import argparse
import pickle
import time
from pathlib import Path

from tokenizers import Tokenizer
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data-directory", type=str, default="ao3_small_dataset/")
    parser.add_argument("--input-data-extension", type=str, default=".md")
    parser.add_argument("--output-data-extension", type=str, default=".tok")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("byte_tokenized_16k.json")

    texts = []

    files = list(Path(args.input_data_directory).rglob("*"+args.input_data_extension))

    print("Loading files")
    for file in tqdm(files):
        with open(file, 'r', encoding='utf-8') as reader:
            texts.append("<SOS>" + reader.read() + "<EOS>")

    print("Tokenizing all files")
    start = time.time()
    tokenized = tokenizer.encode_batch(texts)
    end = time.time()
    print("Took", end-start, "to tokenize")

    print("Saving files")
    for file, tokens in tqdm(zip(files, tokenized)):
        file = str(file.absolute()).replace(args.input_data_extension, args.output_data_extension)

        with open(file, 'wb') as writer:
            pickle.dump(tokens.ids, writer)

    print("Complete")
