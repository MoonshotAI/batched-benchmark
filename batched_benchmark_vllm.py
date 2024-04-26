import os
import pathlib
import subprocess
import argparse

import click
import generate_ob_tests
import yaml
from loguru import logger
import json
import pandas as pd
from tabulate import tabulate

import time
from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine

def analyze_results_offline_inference(raw_results_dir, json_output_fname, md_output_fname, csv_output_fname) -> None:
    data = []  # List to hold all the json data
    for filename in os.listdir(raw_results_dir):
        if filename.endswith('.json'):  # Check if the file is a JSON file
            file_path = os.path.join(raw_results_dir, filename)
            with open(file_path, 'r') as file:
                results = json.load(file)
                data.append(results)
    # Create a DataFrame
    df = pd.DataFrame(data)
    # Saving to json
    with open(json_output_fname, 'w') as f:
        json.dump(data, f)
    # Saving to CSV
    df.to_csv(csv_output_fname, index=False)
    # Saving to Markdown
    with open(md_output_fname, 'w') as f:
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

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

def check_step_type(step_outputs) -> str:
    step_type = None
    for output in step_outputs:
        cur_type = "prefill" if len(output.outputs[0].token_ids) == 1 else "decode"
        if step_type is None:
            step_type = cur_type
        elif step_type != "mixed" and step_type != cur_type:
            logger.warn("Mixed prefill and decode scheduling. Ignored.")
            step_type = "mixed"
    return step_type

def start_vllm(engine_args) -> LLMEngine:
    logger.info("Start vllm init...")
    start = time.time()
    llm = LLMEngine.from_engine_args(engine_args)
    end = time.time()
    logger.info("vllm initialized... Time elapsed:", end - start, "seconds")
    return llm

def start_test(llm, num_requests, n_batch, prompt_tokens, prompt_fname, output_tokens, output_fname) -> None:
    logger.info(f"Start test with {num_requests} requests, n_batch {n_batch}, prompt_tokens {prompt_tokens}, prompt_fname {prompt_fname}, output_tokens {output_tokens}, output_fname {output_fname}")
    request_id = 0
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=output_tokens)
    prompts = []
    with open(prompt_fname, 'r') as file:
        for line in file:
            json_data = json.loads(line.strip())
            prompts.append(json_data["prompt"])
    assert len(prompts) >= num_requests
    prefilling_times = []
    prefilling_nbatch = []
    decode_tokens = 0.0
    decode_time = 0.0
    decode_nbatch = []
    warning_flag = False
    while request_id < num_requests:
        cur_request_id = []
        for _ in range(n_batch):
            llm.add_request(request_id=request_id, prompt=prompts[request_id], sampling_params=sampling_params)
            cur_request_id.append(request_id)
            request_id += 1
            if request_id >= num_requests:
                break
        outputs = []
        step = 0
        while llm.has_unfinished_requests():
            start = time.perf_counter()
            step_outputs = llm.step()
            end = time.perf_counter()
            step_type = check_step_type(step_outputs)
            # print(f"step: {step} n_output: {len(step_outputs)} type {step_type}")
            # for id, output in enumerate(step_outputs):
            #     print(f"id:{id} {output}")
            # print('time used : {}'.format(end - start))
            time_used = end - start
            cur_nbatch = len(step_outputs)
            if step_type == "prefill":
                prefilling_times.append(time_used)
                prefilling_nbatch.append(cur_nbatch)
                if cur_nbatch != n_batch:
                    logger.warning('real prefilling bs {} less than input bs {}'.format(cur_nbatch, n_batch))
            elif step_type == "decode":
                decode_tokens += cur_nbatch
                decode_nbatch.append(cur_nbatch)
                if not warning_flag and cur_nbatch != n_batch:
                    logger.warning('real decode bs {} less than input bs {}'.format(cur_nbatch, n_batch))
                    warning_flag = True
                decode_time += time_used
            step += 1
        logger.info(f"Finished request {cur_request_id}. Current batch: Prefilling time: {prefilling_times[-1]}, overall decode tps: {decode_tokens / decode_time}")
    prefilling_time_avg = sum(prefilling_times) / sum(prefilling_nbatch)
    prefilling_bs = 1.0 * sum(prefilling_nbatch) / len(prefilling_nbatch)
    prefilling_tps = prompt_tokens / prefilling_time_avg
    decode_tps = decode_tokens / decode_time
    decode_bs = 1.0 * sum(decode_nbatch) / len(decode_nbatch)
    logger.info(f"prompt tokens: {prompt_tokens}, n_batch: {n_batch}, average prefilling time: {prefilling_time_avg}, prefilling bs: {prefilling_bs}, prefilling tps: {prefilling_tps}, total decode time: {decode_time}, decode bs: {decode_bs}, decode tps: {decode_tps}")
    results = {}
    results["prompt_tokens"] = prompt_tokens
    results["n_batch"] = n_batch
    results["prefilling_time_avg"] = prefilling_time_avg
    results["prefilling_tps"] = prefilling_tps
    results["decode_time"] = decode_time
    results["decode_tps"] = decode_tps
    results["prefilling_bs"] = prefilling_bs
    results["decode_bs"] = decode_bs
    json.dump(results, open(output_fname, "w"))

@click.command(context_settings={"show_default": True, "ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--config", "-c", type=str, help="path to yaml config for tests",
              required=True)
@click.option("--prompt-dir", "-p", type=str,
              help="dir containing prompt files (*.jsonl). If not set or some test cases"
              "are not satisfied, we use tokenizer and dataset to generate.")
@click.option("--tokenizer", "-t", type=str,
              help="tokenizer model for generating prompts")
@click.option("--dataset", "-d", type=str,
              help="dataset for generating prompts")
@click.option("--output-dir", "-o", type=str,
              help="output dir (also for intermediate results)", required=True)
@click.option("--gpu", "-g", type=str, default=None)
@click.pass_context
def main(ctx, config: str, prompt_dir: str,
         tokenizer: str, dataset: str, output_dir: str,
         gpu: str
         ):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    gpu_num = len(gpu.split(","))
    logger.info(f"set visible gpus: {gpu}, gpu num: {gpu_num}")
    logger.info(f"output dir: {output_dir}")
    # we pass the rest args to vllm engine. A little bit hacky
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    cli_args = parser.parse_args(ctx.args)
    engine_args = EngineArgs.from_cli_args(cli_args)
    logger.info(f"engine_args: {engine_args}")

    # validate engine args
    assert engine_args.tensor_parallel_size == gpu_num, f"tensor parallel size {engine_args.tensor_parallel_size} != gpu num {gpu_num}"
    max_model_len = engine_args.max_model_len

    vllm = start_vllm(engine_args)

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
    # env for ob
    new_env = os.environ | {"ENABLE_GZIP_PROVIDER_REQUEST": "1"}
    for test in tests:
        prompt_tokens = test["prompt_tokens"]
        output_tokens = test["output_tokens"]
        concurrency = test["concurrency"]
        num_requests = test["num_request"]
        if prompt_tokens >= max_model_len:
            logger.warning(f"prompt tokens {prompt_tokens} exceeds max model len {max_model_len}. Ignore test case.")
            continue
        prompt_fname = str(prompt_fname_mapping[prompt_tokens])
        output_fname = raw_results_dir.joinpath(
            f"p{prompt_tokens}_c{concurrency}.json")
        start_test(vllm, num_requests, concurrency, prompt_tokens, prompt_fname, output_tokens, output_fname)

    logger.info("tests finished")
    logger.info("processing raw results")
    # md for human and json for pandas
    final_result_md = final_results_dir.joinpath("summary.md")
    final_result_csv = final_results_dir.joinpath("summary.csv")
    final_result_json = final_results_dir.joinpath("summary.json")
    analyze_results_offline_inference(str(raw_results_dir), str(final_result_json), str(final_result_md), str(final_result_csv))
    logger.info(f"all finished, result: {final_result_md} and {final_result_csv}")


if __name__ == "__main__":
    main()
