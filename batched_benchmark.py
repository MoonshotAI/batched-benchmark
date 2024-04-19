import os
import pathlib
import subprocess

import analyze_result
import click
import generate_ob_tests
import yaml
import jsonlines
from loguru import logger


def analyze_tests(raw_tests: dict) -> (list, list):
    default_configs = raw_tests["default_configs"]
    tests = []
    # prompt_len -> num of prompts
    prompts = {}
    for test in raw_tests["tests"]:
        prompt_tokens = test["prompt_tokens"]
        assert prompt_tokens not in prompts
        # num of prompts to generate
        num = 0
        output_tokens = test.get(
            "output_tokens", default_configs["output_tokens"])
        requests = test.get("requests", default_configs["requests"])
        for request in requests:
            tests.append({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "concurrency": request["concurrency"],
                "num_request": request["num_request"],
            })
            num = max(num, request["num_request"])
        num = min(num, 1000)
        assert num > 0
        prompts[prompt_tokens] = num
    return prompts, tests


@click.command(context_settings={"show_default": True})
@click.option("--executable", "-e", type=str, help="path to ob executbale",
              default="ob")
@click.option("--config", "-c", type=str, help="path to yaml config for tests",
              required=True)
@click.option("--service", "-s", type=str, help="vllm service",
              default="http://localhost:8888/v1")
@click.option("--prompt-dir", "-p", type=str,
              help="dir containing prompt files (*.jsonl). If not set or some test cases"
              "are not satisfied, we use tokenizer and dataset to generate.")
@click.option("--model", "-m", type=str, default="", help="model name")
@click.option("--tokenizer", "-t", type=str,
              help="tokenizer model for generating prompts")
@click.option("--dataset", "-d", type=str,
              help="dataset for generating prompts")
@click.option("--output-dir", "-o", type=str,
              help="output dir (also for intermediate results)", required=True)
@click.option("--dry-run", is_flag=True, default=False,
              help="show commands without running")
@click.option("--disable-gzip", is_flag=True, default=False,
              help="do not use gzip for requests")
def main(executable: str, config: str, service: str, prompt_dir: str, model: str,
         tokenizer: str, dataset: str, output_dir: str, dry_run: bool,
         disable_gzip: bool):
    with open(config, "r") as f:
        raw_tests = yaml.safe_load(f)
    prompts, tests = analyze_tests(raw_tests)
    try:
        pathlib.Path(output_dir).mkdir(parents=True)
    except:
        logger.error(f"output dir {output_dir} exists")
        exit(1)
    raw_results_dir = pathlib.Path(output_dir).joinpath("raw_results")
    final_results_dir = pathlib.Path(output_dir).joinpath("final_results")
    raw_results_dir.mkdir()
    final_results_dir.mkdir()
    # len -> fname
    prompt_fname_mapping = {}
    # load provided prompts first
    if prompt_dir is not None and pathlib.Path(prompt_dir).is_dir():
        logger.info("loading provided prompts")
        provided_prompt_dir = pathlib.Path(prompt_dir)
        # Note: we check on name now
        for prompt_fname in provided_prompt_dir.glob("*.jsonl"):
            prompt_len = int(prompt_fname.stem)
            if prompt_len in prompts:
                with open(prompt_fname) as fin:
                    lines = sum(1 for _ in fin)
                if lines < prompts[prompt_len]:
                    logger.info(f"the lines of provided prompt file {prompt_fname} is less than the benchmark needs")
                else:
                    prompt_fname_mapping[prompt_len] = prompt_fname
                    del prompts[prompt_len]
                    logger.info(f"loaded {prompt_len}:{prompt_fname}")
    # generate rest prompts
    if len(prompts) > 0:
        logger.info("generating prompts")
        output_prompt_dir = pathlib.Path(output_dir).joinpath("prompts")
        output_prompt_dir.mkdir()
        for prompt_len in prompts:
            prompt_file = str(output_prompt_dir.joinpath(f"{prompt_len}.jsonl"))
            prompt_fname_mapping[prompt_len] = prompt_file
            generate_ob_tests.main([
                "--dataset", dataset,
                "--tokenizer", tokenizer,
                "--min-tokens", prompt_len,
                "--max-tokens", prompt_len,
                "--count", prompts[prompt_len],
                "--output", prompt_file,
            ], standalone_mode=False)
        logger.info("all prompts generated")
    logger.info("running tests")
    for test in tests:
        prompt_tokens = test["prompt_tokens"]
        output_tokens = test["output_tokens"]
        concurrency = test["concurrency"]
        num_requests = test["num_request"]
        prompt_fname = str(prompt_fname_mapping[prompt_tokens])
        output_fname = raw_results_dir.joinpath(
            f"p{prompt_tokens}_c{concurrency}.json")
        dump_fname = raw_results_dir.joinpath(
            f"dump_p{prompt_tokens}_c{concurrency}.jsonl"
        )
        command = [executable,
                   "-e", service,
                   "-i", prompt_fname,
                   "-n", str(num_requests),
                   "-c", str(concurrency),
                   "--max-tokens", str(output_tokens),
                   "--temperature", "0.3",
                   "--format", "json",
                   "--model", model,
                   "--dump-output", dump_fname,
                   "--ignore-eos"]
        logger.info(f"cmd: {command}")
        if not dry_run:
            subprocess.check_call(command, stdout=open(
                output_fname, "w"))

    if dry_run:
        return
    logger.info("tests finished")
    logger.info("processing raw results")
    # md for human and json for pandas
    final_result_md = final_results_dir.joinpath("summary.md")
    final_result_json = final_results_dir.joinpath("summary.json")
    analyze_result.main([
        "--dir", str(raw_results_dir),
        "--output", str(final_result_md),
        "--format", "markdown",
        "--verbose",
    ], standalone_mode=False)
    analyze_result.main([
        "--dir", str(raw_results_dir),
        "--output", str(final_result_json),
        "--format", "json",
    ], standalone_mode=False)
    logger.info(f"all finished, result: {final_result_md} and {final_result_json}")


if __name__ == "__main__":
    main()
