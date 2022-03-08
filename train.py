import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import LongTensor
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TextDataset
from model import Encoder


def infinite_dataloader(dataloader: DataLoader) -> torch.LongTensor:
    while True:
        for batch in dataloader:
            yield batch


def build_casual_mask(context_length: int) -> torch.BoolTensor:
    mask = torch.tril(torch.ones(context_length, context_length))
    # mask.shape = [train_ctx_len, train_ctx_len]
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape = [batch_size, 1, train_ctx_len, train_ctx_len]
    return mask.bool()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size",
                        choices=["xsmall", "small", "medium", "large", "xlarge"], type=str,
                        default="xsmall", help="model sizes")

    parser.add_argument("--max-ctx-len", type=int, default=1024)
    parser.add_argument("--train-ctx-len", type=int, default=128)
    parser.add_argument("--test-ctx-lens", type=list, default=[128, 2048])
    parser.add_argument("--position-start-augmentation", type=bool, default=False)

    parser.add_argument("--abs-pos-embed", choices=["sinusoidal", "scaled_sinusoidal", "learned", "none"],
                        type=str, default="none")
    parser.add_argument("--rel-pos-embed",
                        choices=["linear_cpb", "log_cpb", "fourier_cpb", "alibi", "rotary", "none", "linear_cpb_large"])

    parser.add_argument("--num-train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clipping", type=float, default=1.0)

    parser.add_argument("--ckpt", type=str, default="ckpt.pt")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.model_size == "xsmall":  # 3.2M non embedding parameters
        n_layers = 4
        width = 256
        n_heads = 4
    elif args.model_size == "small":  # 10.6M non embedding parameters
        n_layers = 6
        width = 384
        n_heads = 6
    elif args.model_size == "medium":  # 25.2M non embedding parameters
        n_layers = 8
        width = 512
        n_heads = 8
    elif args.model_size == "large":  # 85.1M non embedding parameters
        n_layers = 12
        width = 768
        n_heads = 12
    elif args.model_size == "xlarge":  # 201.5M non embedding parameters
        n_layers = 16
        width = 1024
        n_heads = 16
    else:
        raise Exception("invalid model choice")

    # apply random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # if using relative pos embeddings
    # and the absolute embeddings are the kind that can be length adjusted after training (none/sinusoidal)
    # then set model max context length to training length
    if args.rel_pos_embed != "none" and args.abs_pos_embed in ["none", "sinusoidal", "scaled_sinusoidal"]:
        model = Encoder(8192, width, n_layers, n_heads, width * 4, 0.1, torch.device("cuda"), args.train_ctx_len,
                        args.abs_pos_embed, args.rel_pos_embed).cuda()
    # otherwise set max context length to max context length if no relative pos embeddings
    elif args.rel_pos_embed == "none":
        model = Encoder(8192, width, n_layers, n_heads, width * 4, 0.1, torch.device("cuda"), args.max_ctx_len,
                        args.abs_pos_embed, args.rel_pos_embed).cuda()
    # learned pos embeddings + relative pos embeddings don't mix
    else:
        raise ValueError("Cannot have both learned absolute embeddings and relative embeddings (for now)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embedding_params = sum(
        p.numel() for n, p in model.named_parameters() if
        p.requires_grad and ("embedding" not in n and "fc_out" not in n))

    print("Initialized transformer model with", total_params, "total parameters")
    print("Total non-embedding parameters:", non_embedding_params)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    train_dataset = TextDataset(list(Path("ao3_small_dataset/train").rglob("*.tok")), "byte_tokenized_8k.json",
                                args.train_ctx_len, args.train_ctx_len, pretokenized=True)

    # less than 1 epoch is trained, so to ensure all models see the same data in the same order, shuffling is turned off
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                  drop_last=True)

    train_inf_loader: LongTensor = infinite_dataloader(train_dataloader)

    cross_entropy = torch.nn.CrossEntropyLoss()

    scaler = GradScaler()
    for step in tqdm(range(args.num_train_steps)):
        optimizer.zero_grad(set_to_none=True)

        batch = next(train_inf_loader).cuda()

        if args.position_start_augmentation:
            start = torch.randint(0, args.max_ctx_len - args.train_ctx_len, (args.batch_size,))
        else:
            start = torch.zeros(args.batch_size)

        start = start.unsqueeze(1).cuda()

        positions = torch.arange(0, args.train_ctx_len).unsqueeze(0).repeat(args.batch_size, 1).cuda() + start
        # positions.shape = [batch_size, train_ctx_len]

        mask = build_casual_mask(args.train_ctx_len - 1).cuda()
        # mask.shape = [batch_size, 1, train_ctx_len, train_ctx_len]
        with autocast():
            output = model(batch[:, :-1], mask, positions[:, :-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            targets = batch[:, 1:].contiguous().view(-1)

            loss = cross_entropy(output, targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping)

        scaler.step(optimizer)

        scaler.update()

        if (step + 1) % 100 == 0:
            print(f"Step: {step + 1}\t Loss: {loss.item():.3f}")

    valid_datasets = [
        TextDataset(list(Path("ao3_small_dataset/valid").rglob("*.tok")), "byte_tokenized_8k.json", test_length,
                    stride=test_length, pretokenized=True) for test_length in args.test_ctx_lens]

    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    all_losses = []

    avg_losses = []
    median_losses = []

    max_length_and_batch = 128 * 128 * 2

    # get rid of adam??
    optimizer = None
    with torch.inference_mode():
        for valid_dataset, test_length in zip(valid_datasets, args.test_ctx_lens):
            losses = np.array(np.zeros((valid_dataset.length, test_length - 1)))

            current_idx = 0

            valid_dataloader = DataLoader(valid_dataset, batch_size=max_length_and_batch // test_length, shuffle=False,
                                          num_workers=2)

            if args.rel_pos_embed != "none" and args.abs_pos_embed in ["none", "sinusoidal", "scaled_sinusoidal"]:
                model.update_sizes(test_length)

            for batch in tqdm(valid_dataloader):
                batch = batch.cuda()

                batch_size = batch.shape[0]

                positions = torch.arange(0, test_length).unsqueeze(0).repeat(batch_size, 1).cuda()
                # positions.shape = [batch_size, test_length]

                mask = build_casual_mask(test_length - 1).cuda()
                # mask.shape = [batch_size, 1, test_length-1, test_length-1]

                with autocast():
                    output = model(batch[:, :-1], mask, positions[:, :-1])

                    output_dim = output.shape[-1]

                    output = output.contiguous().view(-1, output_dim)
                    targets = batch[:, 1:].contiguous().view(-1)

                    loss = cross_entropy(output, targets)

                loss = loss.view(batch_size, -1)

                losses[current_idx:current_idx + batch_size, :] = loss.cpu().numpy()
                current_idx += batch_size

            avg_loss = np.mean(losses)
            median_loss = np.median(losses)

            avg_losses.append(avg_loss)
            median_losses.append(median_loss)
            all_losses.append(losses)

    print()
    for test_length, avg_loss in zip(args.test_ctx_lens, avg_losses):
        print(
            f"Test Length: {test_length}\t\tAvg Test Loss: \t\t{avg_loss:.5f}\t\t Avg Test Perplexity: \t\t{math.exp(avg_loss):.5f}")

    print()
    for test_length, median_loss in zip(args.test_ctx_lens, median_losses):
        print(
            f"Test Length: {test_length}\t\tMedian Test Loss: \t{median_loss:.5f}\t\t Median Test Perplexity: \t{math.exp(median_loss):.5f}")

    with open(args.ckpt.replace(".pt", "_all_losses.npy"), 'wb') as writer:
        pickle.dump(losses, writer)

    torch.save(model.state_dict(), args.ckpt)
