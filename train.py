import torch.optim as optim
import torch

from torch.utils.data import DataLoader

import argparse

from model import Encoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model sizes based on "The Depth-to-Width interplay in Self-Attention", https://arxiv.org/abs/2006.12467
    # specifically Figure 3(c) and Appendix D Table 4(a)
    parser.add_argument("--model-size",
                        choices=["xsmall", "small", "medium", "large", "xlarge"], type=str,
                        default="xlarge", help="model sizes")

    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument("--train-context-length", type=int, default=128)
    parser.add_argument("--position-start-augmentation", type=bool, default=True)

    parser.add_argument("--absolute-position-embedding", choices=["sinusoidal", "scaled_sinusoidal", "learned", "none"],
                        type=str, default="none")

    args = parser.parse_args()

    if args.model_size == "xsmall":  # 1.78M non embedding parameters
        n_layers = 6
        width = 192
        n_heads = 3
    elif args.model_size == "small":  # 6.3M non embedding parameters
        n_layers = 12
        width = 256
        n_heads = 4
    elif args.model_size == "medium":  # 21.3M non embedding parameters
        n_layers = 18
        width = 384
        n_heads = 6
    elif args.model_size == "large":  # 50.4M non embedding parameters
        n_layers = 24
        width = 512
        n_heads = 8
    elif args.model_size == "xlarge":  # 141.8M non embedding parameters
        n_layers = 30
        width = 768
        n_heads = 12
    else:
        raise Exception("invalid model choice")

    model = Encoder(16384, width, n_layers, n_heads, width * 2, 0.1, torch.device("cuda"), args.max_context_length,
                    args.absolute_position_embedding).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embedding_params = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "embedding" not in n)

    print("Initialized transformer model with", total_params, "total parameters")
    print("Total non-embedding parameters:", non_embedding_params)

    optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
