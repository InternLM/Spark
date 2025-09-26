import itertools
import time
import mmengine
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            prompts = data.get('prompts')
            queries = data.get("query")
            labels = data.get("labels")
            assert len(prompts) == len(queries), "Mismatched input lengths"

            def chunk_data(data_list, num_chunks):
                """Evenly split a list into num_chunks sublists"""
                k, m = divmod(len(data_list), num_chunks)
                return [data_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]

            num_threads = 8
            prompt_chunks = chunk_data(prompts, num_threads)
            query_chunks = chunk_data(queries, num_threads)
            labels_chunks = chunk_data(labels, num_threads)

            def send_request(url, prompts, queries, labels, headers, score_key):
                try:
                    start_time = time.time()
                    payload = {'prompts': prompts, 'query': queries, 'labels': labels}
                    response = requests.post(url=url, json=payload, headers=headers, timeout=180)
                    response.raise_for_status()
                    elapsed_time = time.time() - start_time
                    logger.info(f"Request to {url} took {elapsed_time:.3f} seconds.")
                    result = response.json()
                    assert score_key in result, f"{score_key} not in {result}"
                    return result.get(score_key)
                except Exception as e:
                    logger.error(f"Error with URL {url}: {e}")
                    return None

            results = [None] * num_threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_index = {
                    executor.submit(
                        send_request,
                        url=url.replace('8888', str(8888 + i)),
                        prompts=prompt_chunks[i],
                        queries=query_chunks[i],
                        labels=labels_chunks[i],
                        headers=headers,
                        score_key=score_key
                    ): i for i in range(num_threads)
                }

                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    results[i] = future.result()
                results = list(itertools.chain.from_iterable(results))
                return results

        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, prompts, labels, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries, "prompts": prompts, "labels": labels}, score_key)
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, score_key="rewards"):
    return remote_rm_fn(api_url, queries, prompts, labels, score_key)


if __name__ == "__main__":
    # test utils
    url = "http://172.30.12.180:8888/get_reward"
    data = mmengine.load('tmp.pkl')
    score = remote_rm_fn(url, data['query'], data['prompts'], None)
    print(score)