import argparse
import math
import pickle
from pathlib import Path

import numpy as np
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
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape = [batch_size, 1, train_context_length, train_context_length]
    return mask.bool()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size",
                        choices=["xsmall", "small", "medium", "large", "xlarge"], type=str,
                        default="xsmall", help="model sizes")

    parser.add_argument("--max-context-length", type=int, default=1024)
    parser.add_argument("--train-context-length", type=int, default=128)
    parser.add_argument("--test-context-lengths", type=list, default=[128, 144, 160, 192, 256, 384, 512, 1024])
    parser.add_argument("--position-start-augmentation", type=bool, default=False)

    parser.add_argument("--absolute-position-embedding", choices=["sinusoidal", "scaled_sinusoidal", "learned", "none"],
                        type=str, default="none")
    parser.add_argument("--relative-position-embedding", choices=["linear_cpb", "log_cpb", "alibi", "rotary", "none"])

    parser.add_argument("--num-train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clipping", type=float, default=1.0)

    parser.add_argument("--ckpt", type=str, default="ckpt.pt")

    args = parser.parse_args()

    if args.model_size == "xsmall":  # 7.1M non embedding parameters
        n_layers = 6
        width = 384
        n_heads = 6
    elif args.model_size == "small":  # 16.8M non embedding parameters
        n_layers = 8
        width = 512
        n_heads = 12
    elif args.model_size == "medium":  # 56.7M non embedding parameters
        n_layers = 12
        width = 768
        n_heads = 12
    elif args.model_size == "large":  # 134.3M non embedding parameters
        n_layers = 16
        width = 1024
        n_heads = 16
    elif args.model_size == "xlarge":  # 262.4M non embedding parameters
        n_layers = 20
        width = 1280
        n_heads = 20
    else:
        raise Exception("invalid model choice")

    # if using relative pos embeddings
    # and the absolute embeddings are the kind that can be length adjusted after training (none/sinusoidal)
    # then set model max context length to training length
    if args.relative_position_embedding != "none" and args.absolute_position_embedding in ["none", "sinusoidal", "scaled_sinusoidal"]:
        model = Encoder(8192, width, n_layers, n_heads, width * 2, 0.1, torch.device("cuda"), args.train_context_length,
                        args.absolute_position_embedding, args.relative_position_embedding).cuda()
    # otherwise set max context length to max context length if no relative pos embeddings
    elif args.relative_position_embedding == "none":
        model = Encoder(8192, width, n_layers, n_heads, width * 2, 0.1, torch.device("cuda"), args.max_context_length,
                        args.absolute_position_embedding, args.relative_position_embedding).cuda()
    # learned pos embeddings + relative pos embeddings don't mix
    else:
        raise ValueError("Cannot have both learned absolute embeddings and relative embeddings (for now)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embedding_params = sum(
        p.numel() for n, p in model.named_parameters() if
        p.requires_grad and ("embedding" not in n and "fc_out" not in n))

    print("Initialized transformer model with", total_params, "total parameters")
    print("Total non-embedding parameters:", non_embedding_params)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, eps=1e-8)

    train_dataset = TextDataset(list(Path("ao3_small_dataset/train").rglob("*.tok")), "byte_tokenized_8k.json",
                                args.train_context_length, args.train_context_length, pretokenized=True)

    # less than 1 epoch is trained, so to ensure all models see the same data in the same order, shuffling is turned off
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
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

        output = model(batch[:, :-1], mask, positions[:, :-1])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        targets = batch[:, 1:].contiguous().view(-1)

        loss = cross_entropy(output, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clipping)

        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Step: {step + 1}\t Loss: {loss.item():.3f}")

    valid_datasets = [
        TextDataset(list(Path("ao3_small_dataset/valid").rglob("*.tok")), "byte_tokenized_8k.json", test_length,
                    stride=test_length, pretokenized=True) for test_length in args.test_context_lengths]

    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    all_losses = []

    avg_losses = []
    median_losses = []
    with torch.inference_mode():
        for valid_dataset, test_length in zip(valid_datasets, args.test_context_lengths):
            losses = np.array(np.zeros((valid_dataset.length, test_length-1)))

            current_idx = 0

            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            if args.relative_position_embedding != "none" and args.absolute_position_embedding in ["none", "sinusoidal",
                                                                                                   "scaled_sinusoidal"]:
                model.update_sizes(test_length)

            for batch in tqdm(valid_dataloader):
                batch = batch.cuda()

                batch_size = batch.shape[0]

                positions = torch.arange(0, test_length).unsqueeze(0).repeat(batch_size, 1).cuda()
                # positions.shape = [batch_size, test_length]

                mask = build_casual_mask(test_length - 1, batch_size).cuda()
                # mask.shape = [batch_size, 1, test_length-1, test_length-1]

                output = model(batch[:, :-1], mask, positions[:, :-1])

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                targets = batch[:, 1:].contiguous().view(-1)

                loss = cross_entropy(output, targets)

                loss = loss.view(batch_size, -1)

                losses[current_idx:current_idx+batch_size, :] = loss.cpu().numpy()
                current_idx += batch_size

            avg_loss = np.mean(losses)
            median_loss = np.median(losses)

            avg_losses.append(avg_loss)
            median_losses.append(median_loss)
            all_losses.append(losses)

    print()
    for test_length, avg_loss in zip(args.test_context_lengths, avg_losses):
        print(f"Test Length: {test_length}\t\tAvg Test Loss: \t\t{avg_loss:.5f}\t\t Avg Test Perplexity: \t\t{math.exp(avg_loss):.5f}")

    print()
    for test_length, median_loss in zip(args.test_context_lengths, median_losses):
        print(f"Test Length: {test_length}\t\tMedian Test Loss: \t{median_loss:.5f}\t\t Median Test Perplexity: \t{math.exp(median_loss):.5f}")

    with open(args.ckpt.replace(".pt", "_all_losses.npy"), 'wb') as writer:
        pickle.dump(losses, writer)

    torch.save(model.state_dict(), args.ckpt)
