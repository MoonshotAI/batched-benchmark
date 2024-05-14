import functools
import glob
import json
import os
import re

import click
from tabulate import tabulate
import pandas as pd


def to_int(s: str):
    if s.endswith("k"):
        return int(s.split("k")[0]) * 1024
    return int(s)


def cmp(lhs: dict, rhs: dict):
    if lhs["prompt"] != rhs["prompt"]:
        return lhs["prompt"] - rhs["prompt"]
    return lhs["concurrency"] - rhs["concurrency"]


def load_table(dir: str, verbose: bool=False):
    table = []
    files = glob.glob(dir + "/*.json")
    prog = re.compile(r"p(?P<prompt>\d+)_c(?P<concurrency>\d+).json")
    for fname in files:
        try:
            with open(fname, "r") as f:
                result = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"error loading {fname}")
            continue
        # TODO: this is too tricky
        short_fname = os.path.basename(fname)
        m = prog.match(short_fname)
        prompt = m.group("prompt")
        concurrency = m.group("concurrency")
        n_requests = result["num_failed"] + result["num_succeeded"]
        global_io_token_rate = result["global_token_rate"] + to_int(prompt) * n_requests / result["total_time_elapsed"]
        entry = {
            "prompt": to_int(prompt),
            "concurrency": to_int(concurrency),
            "initial_time_cost": result["initial_time_cost"]["p50"],
            "token_per_second": result["token_per_second"]["p50"],
            "global_token_rate": result["global_token_rate"],
            "num_failed": result["num_failed"],
            "global_io_token_rate": global_io_token_rate,
        }
        if result["num_failed"] > 0 and verbose:
            print(f"WARN: fname={fname} num_failed={result['num_failed']}")
        table.append(entry)
    table.sort(key=functools.cmp_to_key(cmp))
    return table


@click.command(context_settings={"show_default": True})
@click.option("--dir", "-d", type=str, help="dir for bmk results",
              required=True)
@click.option("--output", "-o", type=str, help="output file name",
              required=True)
@click.option("--format", "-f", type=click.Choice(["markdown", "json"]), 
              help="output format", default="markdown")
@click.option("--verbose", is_flag=True, default=False,
              help="verbose")
def main(dir: str, output: str, format: str, verbose: bool):
    table = load_table(dir, verbose=verbose)
    if format == "markdown":
        # mainly for human
        msg = f"# result for {dir}\n"
        headers = ["prompt", "concurrency", "initial_time_cost",
                   "token_per_second", "global_token_rate", "global_io_token_rate"]
        assert set(table[0].keys()).issuperset(set(headers))
        column_names = ["prompt", "concurrency",
                        "prefill (s)", "token/s", "global token/s", "global io token/s"]
        data = [[entry[v] for v in headers] for entry in table]
        if verbose:
            print(tabulate(data, headers, tablefmt="github"))

        with open(output, "w") as f:
            f.write(msg)
            f.write(tabulate(data, column_names, tablefmt="github"))
    else:
        # intermediate format for pandas
        assert format == "json"
        df = pd.DataFrame(table)
        df.to_json(output)


if __name__ == "__main__":
    main()
