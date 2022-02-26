import argparse
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch import LongTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TextDataset
from model import Encoder


def infinite_dataloader(dataloader: DataLoader) -> torch.LongTensor:
    while True:
        for batch in dataloader:
            yield batch


def build_casual_mask(context_length: int, batch_size: int) -> torch.BoolTensor:
    mask = torch.tril(torch.ones(context_length, context_length))
    # mask.shape = [train_context_length, train_context_length]
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    # mask.shape = [batch_size, 1, train_context_length, train_context_length]
    return mask.bool()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size",
                        choices=["xsmall", "small", "medium", "large", "xlarge"], type=str,
                        default="xsmall", help="model sizes")

    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument("--train-context-length", type=int, default=128)
    parser.add_argument("--test-context-lengths", type=list, default=[128, 144, 160, 192, 256, 384, 512])
    parser.add_argument("--position-start-augmentation", type=bool, default=True)

    parser.add_argument("--absolute-position-embedding", choices=["sinusoidal", "scaled_sinusoidal", "learned", "none"],
                        type=str, default="none")

    parser.add_argument("--num-train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clipping", type=float, default=1.0)

    args = parser.parse_args()

    if args.model_size == "xsmall":  # 16.8M non embedding parameters
        n_layers = 8
        width = 512
        n_heads = 8
    elif args.model_size == "small":  # 56.7M non embedding parameters
        n_layers = 12
        width = 768
        n_heads = 12
    elif args.model_size == "medium":  # 134.3M non embedding parameters
        n_layers = 16
        width = 1024
        n_heads = 16
    elif args.model_size == "large":  # 262.4M non embedding parameters
        n_layers = 20
        width = 1280
        n_heads = 20
    elif args.model_size == "xlarge": # 453.3M non embedding parameters
        n_layers = 24
        width = 1536
        n_heads = 24
    else:
        raise Exception("invalid model choice")

    model = Encoder(16384, width, n_layers, n_heads, width * 2, 0.1, torch.device("cuda"), args.max_context_length,
                    args.absolute_position_embedding).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embedding_params = sum(
        p.numel() for n, p in model.named_parameters() if
        p.requires_grad and ("embedding" not in n and "fc_out" not in n))

    print("Initialized transformer model with", total_params, "total parameters")
    print("Total non-embedding parameters:", non_embedding_params)

    optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    train_dataset = TextDataset(list(Path("ao3_small_dataset/train").rglob("*.tok")), "byte_tokenized_16k.json",
                                args.train_context_length, args.train_context_length, pretokenized=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                  drop_last=True)

    train_inf_loader: LongTensor = infinite_dataloader(train_dataloader)

    cross_entropy = torch.nn.CrossEntropyLoss()

    for step in tqdm(range(args.num_train_steps)):
        optimizer.zero_grad(set_to_none=True)

        batch = next(train_inf_loader).cuda()

        if args.position_start_augmentation:
            start = torch.randint(0, args.max_context_length - args.train_context_length, (args.batch_size,))
        else:
            start = torch.zeros(args.batch_size)

        start = start.unsqueeze(1).cuda()

        positions = torch.arange(0, args.train_context_length).unsqueeze(0).repeat(args.batch_size, 1).cuda() + start
        # positions.shape = [batch_size, train_context_length]

        mask = build_casual_mask(args.train_context_length - 1, args.batch_size).cuda()
        # mask.shape = [batch_size, 1, train_context_length, train_context_length]

        output = model(batch[:, :-1], mask)

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        targets = batch[:, 1:].contiguous().view(-1)

        loss = cross_entropy(output, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clipping)

        optimizer.step()

        if step % 100 == 0:
            print(f"Step: {step}\t Loss: {loss.item():.3f}")

    valid_datasets = [
        TextDataset(list(Path("ao3_small_dataset/valid").rglob("*.tok")), "byte_tokenized_16k.json", test_length,
                    stride=args.test_length, pretokenized=True) for test_length in args.test_context_lengths]

    with torch.inference_mode():
        for valid_dataset, test_length in zip(valid_datasets, args.test_context_lengths):
            total_loss = 0
            total_samples = 0

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
            for batch in tqdm(valid_dataloader):
                batch = batch.cuda()

                batch_size = batch.shape[0]

                positions = torch.arange(0, test_length).unsqueeze(0).repeat(batch_size, 1).cuda()
                # positions.shape = [batch_size, test_length]

                mask = build_casual_mask(test_length - 1, batch_size).cuda()
                # mask.shape = [batch_size, 1, test_length-1, test_length-1]

                output = model(batch[:, :-1], mask)

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                targets = batch[:, 1:].contiguous().view(-1)

                loss = cross_entropy(output, targets)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples
            print()
            print(f"Test Length: {test_length}\t Test Loss: {avg_loss:.5f}\t Test Perplexity: {math.exp(avg_loss):.5f}")
