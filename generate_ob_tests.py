#!/usr/bin/env python3
import click
import json
from loguru import logger
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def create_tokens_stream(dataset, tknz, loop=False):
    with open(dataset) as fin:
        data = json.load(fin)
    logger.debug(f"loaded {len(data)} conversations from dataset {dataset}, loop={loop}")
    while True:
        for conv in data:
            for msg in conv["conversations"]:
                yield from tknz.encode(msg["value"])
        if not loop:
            break


@click.command()
@click.option("--dataset", type=str, help="huggingface dataset")
@click.option("--tokenizer", type=str)
@click.option("--min-tokens", type=int, default=1024)
@click.option("--max-tokens", type=int)
@click.option("--count", type=int)
@click.option("--output", type=str)
def main(dataset: str, tokenizer: str, output: str, min_tokens: int, max_tokens: int, count: int):
    tknz = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    tokens_stream = create_tokens_stream(dataset, tknz, loop=count is not None)

    if max_tokens is None:
        max_tokens = min_tokens
        logger.debug(f"max tokens default as {max_tokens}")
    logger.info(f"start generating {count} samples with {min_tokens} ~ {max_tokens} tokens each")

    if output is None:
        output = BASE_DIR / f"corpora/tokens-{min_tokens}-{max_tokens}.jsonl"
        logger.debug(f"output file default as {output}")
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)

    total_tokens, total_length = 0, 0
    with open(output, "w") as fout:
        for rounds in tqdm(range(count or int(1e9))):
            nr_tokens = random.randint(min_tokens, max_tokens)
            # logger.debug(f"round {rounds} with {nr_tokens} tokens")
            try:
                tokens = [next(tokens_stream) for _ in range(nr_tokens)]
            except StopIteration:
                logger.warning(f"tokens stream exhausted after {rounds} rounds")
                break
            text = tknz.decode(tokens)
            total_tokens += len(tokens)
            total_length += len(text)
            print(json.dumps({"prompt": text}), file=fout)
    logger.info(f"generated {total_tokens} tokens, {total_length} characters, {rounds} samples")

if __name__ == "__main__":
    main()