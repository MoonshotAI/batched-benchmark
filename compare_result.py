import analyze_result
import click
import pandas as pd

# append to msg
def check_failed(row, name: str):
    if row["num_failed"] != 0:
        return (f"WARN: {name} prompt={row['prompt']} concurrency="
                f"{row['concurrency']}, num_failed={row['num_failed']}\n")
    return ""

@click.command(context_settings={"show_default": True})
@click.option("--baseline", "-b", type=str, help="path for raw baseline results",
              required=True)
@click.option("--current", "-c", type=str, help="path for raw current results to be "
              "compared with baseline", required=True)
@click.option("--format", "-f", type=click.Choice(["dir", "json"]),
              help="input format, dir is for raw_results, "
              "and json is for final_results/summary.json",
              default="dir")
@click.option("--output", "-o", type=str, help="output file (markdown)",
              required=True)
def main(baseline: str, current: str, format: str, output: str):
    if format == "dir":
        baseline_df = pd.DataFrame(analyze_result.load_table(baseline))
        current_df = pd.DataFrame(analyze_result.load_table(current))
    else:
        assert format == "json"
        baseline_df = pd.read_json(baseline)
        current_df = pd.read_json(current)
    # for num_failed
    warn_msg = ""
    for _, row in baseline_df.iterrows():
        warn_msg += check_failed(row, "baseline")
    for _, row in current_df.iterrows():
        warn_msg += check_failed(row, "current")
    # drop num_failed
    baseline_df.drop("num_failed", axis=1, inplace=True)
    current_df.drop("num_failed", axis=1, inplace=True)
    common_names = ["prompt", "concurrency"]
    to_compare_names = ["initial_time_cost",
                        "token_per_second", "global_token_rate", "global_io_token_rate"]
    raw_name_mapping = {
        "initial_time_cost": "prefill (s)",
        "token_per_second": "token/s",
        "global_token_rate": "global token/s",
        "global_io_token_rate": "global io token/s",
    }
    assert set(common_names) | set(to_compare_names) == set(baseline_df.columns)
    suffixes = ("_baseline", "_current")
    # TODO: mark missing rows?
    new_df = baseline_df.merge(current_df, on=common_names, suffixes=suffixes)
    for name in to_compare_names:
        diff_name = name + "_diff"
        baseline_name = name + suffixes[0]
        current_name = name + suffixes[1]
        new_df[diff_name] = new_df.apply(lambda row: (
            row[current_name] - row[baseline_name]) / row[baseline_name] * 100 if row[baseline_name] != 0 else None, axis=1)
    # reorder the columns
    new_order = common_names
    for name in to_compare_names:
        new_order.extend((name+suffixes[0], name+suffixes[1], name+"_diff"))
    new_df = new_df.reindex(columns=new_order)
    # rename
    # FIXME: better
    name_mapping = {name+"_diff": "diff(%)" for name in to_compare_names}
    for name in to_compare_names:
        name_mapping[name+suffixes[0]] = "baseline " + raw_name_mapping[name]
        name_mapping[name+suffixes[1]] = "current " + raw_name_mapping[name]
    new_df = new_df.rename(columns=name_mapping)
    print(warn_msg)
    print(new_df)
    with open(output, "w") as f:
        f.write(f"# result for {current} vs {baseline}(baseline)\n")
        f.write(warn_msg.replace("\n", "\n\n"))
        f.write(new_df.to_markdown(tablefmt="github", index=False))

if __name__ == "__main__":
    main()
