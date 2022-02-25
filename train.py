import torch.optim as optim
import torch

from torch.utils.data import DataLoader

import argparse

from model import Encoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size",
                        choices=["2.5M", "4.3M", "10.4M", "21.2M", "63.4M", "143.3M", "273.5M", "466.7M"], type=str,
                        default="4.3M")

    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument("--train-context-length", type=int, default=128)
    parser.add_argument("--position-start-augmentation", type=bool, default=True)

    parser.add_argument("--absolute-position-embedding", choices=["sinusoidal", "scaled_sinusoidal", "learned", "none"],
                        type=str, default="none")

    args = parser.parse_args()

    if args.model_size == "2.5M":
        n_layers = 3
        width = 192
        n_heads = 3
    elif args.model_size == "4.3M":
        n_layers = 4
        width = 256
        n_heads = 4
    elif args.model_size == "10.4M":
        n_layers = 6
        width = 384
        n_heads = 6
    elif args.model_size == "21.2M":
        n_layers = 8
        width = 512
        n_heads = 8
    elif args.model_size == "63.4M":
        n_layers = 12
        width = 768
        n_heads = 12
    elif args.model_size == "143.3M":
        n_layers = 16
        width = 1024
        n_heads = 16
    elif args.model_size == "273.5M":
        n_layers = 20
        width = 1280
        n_heads = 20
    elif args.model_size == "466.7M":
        n_layers = 24
        width = 1536
        n_heads = 24
    else:
        raise Exception("invalid model choice")

    model = Encoder(8192, width, n_layers, n_heads, width * 2, 0.1, torch.device("cuda"), args.max_context_length,
                    args.absolute_position_embedding).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(total_params)

    print("Initialized transformer model with", (total_params // 10_000) / 100, "M parameters")

    optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    


