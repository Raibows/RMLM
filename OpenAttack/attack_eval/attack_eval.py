import sys
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import logging
from tqdm import tqdm
from ..utils import visualizer, result_visualizer, get_language, language_by_name
from .utils import worker_process, worker_init, attack_process
from ..tags import *
from ..text_process.tokenizer import Tokenizer, get_default_tokenizer
from ..victim.base import Victim
from ..attackers.base import Attacker
from ..metric import AttackMetric, MetricSelector

import multiprocessing as mp

logger = logging.getLogger(__name__)

class AttackEval:
    def __init__(self,
        attacker_name : str,
        attacker : Attacker,
        victim : Victim,
        language : Optional[str] = None,
        tokenizer : Optional[Tokenizer] = None,
        invoke_limit : Optional[int] = None,
        metrics : List[Union[AttackMetric, MetricSelector]] = []
    ):
        """
        `AttackEval` is a class used to evaluate attack metrics in OpenAttack.

        Args:
            attacker: An attacker, must be an instance of :py:class:`.Attacker` .
            victim: A victim model, must be an instance of :py:class:`.Vicitm` .
            language: The language used for the evaluation. If is `None` then `AttackEval` will intelligently select the language based on other parameters.
            tokenizer: A tokenizer used for visualization.
            invoke_limit: Limit on the number of model invokes.
            metrics: A list of metrics. Each element must be an instance of :py:class:`.AttackMetric` or :py:class:`.MetricSelector` .

        """
        self.attacker_name = attacker_name
        if language is None:
            lst = [attacker]
            if tokenizer is not None:
                lst.append(tokenizer)
            if victim.supported_language is not None:
                lst.append(victim)
            for it in metrics:
                if isinstance(it, AttackMetric):
                    lst.append(it)

            lang_tag = get_language(lst)
        else:
            lang_tag = language_by_name(language)
            if lang_tag is None:
                raise ValueError("Unsupported language `%s` in attack eval" % language)

        self._tags = { lang_tag }

        if tokenizer is None:
            self.tokenizer = get_default_tokenizer(lang_tag)
        else:
            self.tokenizer = tokenizer

        self.attacker = attacker
        self.victim = victim
        self.metrics = []
        for it in metrics:
            if isinstance(it, MetricSelector):
                v = it.select(lang_tag)
                if v is None:
                    raise RuntimeError("`%s` does not support language %s" % (it.__class__.__name__, lang_tag.name))
                self.metrics.append( v )
            elif isinstance(it, AttackMetric):
                self.metrics.append( it )
            else:
                raise TypeError("`metrics` got %s, expect `MetricSelector` or `AttackMetric`" % it.__class__.__name__)
        self.invoke_limit = invoke_limit
        
    @property
    def TAGS(self):
        return self._tags
    
    def __measure(self, data, adversarial_sample):
        ret = {}
        for it in self.metrics:
            value = it.after_attack(data, adversarial_sample)
            if value is not None:
                ret[it.name] = value
        return ret


    def __iter_dataset(self, dataset):
        for data in dataset:
            v = data
            for it in self.metrics:
                ret = it.before_attack(v)
                if ret is not None:
                    v = ret
            yield v
    
    def __iter_metrics(self, iterable_result):
        for data, result in iterable_result:
            adversarial_sample, attack_time, invoke_times, OOM_FAIL, prob = result
            ret = {
                "data": data,
                "success": adversarial_sample is not None,
                "result": adversarial_sample,
                'oom_fail': OOM_FAIL,
                'prob': prob,
                "metrics": {
                    "Running Time": attack_time,
                    "Query Exceeded": self.invoke_limit is not None and invoke_times > self.invoke_limit,
                    "Victim Model Queries": invoke_times,
                    ** self.__measure(data, adversarial_sample)
                }
            }
            yield ret

    def ieval(self, dataset : Iterable[Dict[str, Any]], num_workers : int = 0, chunk_size : Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Iterable evaluation function of `AttackEval` returns an Iterator of result.

        Args:
            dataset: An iterable dataset.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Yields:
            A dict contains the result of each input samples.

        """

        if num_workers > 0:
            ctx = mp.get_context("spawn")
            if chunk_size is None:
                chunk_size = num_workers
            with ctx.Pool(num_workers, initializer=worker_init, initargs=(self.attacker, self.victim, self.invoke_limit)) as pool:
                for ret in self.__iter_metrics(zip(dataset, pool.imap(worker_process, self.__iter_dataset(dataset), chunksize=chunk_size))):
                    yield ret
                   
        else:
            def result_iter():
                for data in self.__iter_dataset(dataset):
                    yield attack_process(self.attacker, self.victim, data, self.invoke_limit)
            for ret in self.__iter_metrics(zip(dataset, result_iter())):
                yield ret

    def eval(self, dataset: Iterable[Dict[str, Any]], total_len : Optional[int] = None, visualize : bool = False, progress_bar : bool = False, num_workers : int = 0, chunk_size : Optional[int] = None, x_map_prob=None, total_attack_set_num_include_misclsf=-1, tqdm_prefix=None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Returns:
            A dict of attack evaluation summaries.

        """


        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        def tqdm_writer(x):
            return tqdm.write(x, end="")
        
        if progress_bar:
            tqdm_prefix = self.attacker_name if tqdm_prefix is None else tqdm_prefix
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len, desc=tqdm_prefix)
        else:
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0
        oom_fail_inst = 0
        write_results = []
        expect_acc_record = []


        # Begin for
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])  # successful attack
            oom_fail_inst += int(res['oom_fail'])
            ori_x = res["data"]["x"]
            ori_y_prob = x_map_prob[ori_x]
            if visualize and (TAG_Classification in self.victim.TAGS):
                if res["success"]:
                    expect_acc_record.append(False)
                    adv_x = res["result"]
                    adv_y_prob = res['prob']
                    adv_y_pred = adv_y_prob.argmax().item()
                    ori_y_pred = ori_y_prob.argmax().item()
                    assert adv_y_pred != ori_y_pred
                    save_results = {'ori_y': ori_y_pred, 'adv_y': adv_y_pred, 'ori_prob': ori_y_prob.tolist(), 'adv_prob': adv_y_prob.tolist(), 'ori_x': ori_x, 'adv_x': adv_x}
                    save_results.update(res['metrics'])
                    write_results.append(save_results)
                else:
                    expect_acc_record.append(True)
                    adv_x = None
                    adv_y_prob = None

                info = res["metrics"]
                info["Succeed"] = res["success"]
                ASR = success_inst / total_inst
                ACC = (total_inst - success_inst) / total_attack_set_num_include_misclsf

                accumulate_acc_expect = total_len / total_inst * ACC
                last_50_acc = sum(expect_acc_record[-50:]) / total_attack_set_num_include_misclsf
                recent_acc_expect = total_len / min(50, len(expect_acc_record)) * last_50_acc
                expect_acc = max(ACC, 0.6 * accumulate_acc_expect + 0.4 * recent_acc_expect)

                tqdm.set_postfix_str(result_iterator, f'ACC/Expect {ACC:.3f}/{expect_acc:.3f} ASR {ASR:.3f} OOM {oom_fail_inst}')
                if progress_bar:
                    visualizer(i + 1, ori_x, ori_y_prob, adv_x, adv_y_prob, info, tqdm_writer, self.tokenizer)
                else:
                    visualizer(i + 1, ori_x, ori_y_prob, adv_x, adv_y_prob, info, sys.stdout.write, self.tokenizer)

            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)
        # End for

        summary = {}
        summary["Total Attacked Instances"] = total_inst
        summary["Successful Instances"] = success_inst
        summary["Attack Success Rate"] = success_inst / total_inst
        summary['oom_fail'] = oom_fail_inst
        for kw in total_result_cnt.keys():
            if kw in ["Succeed"]:
                continue
            if kw in ["Query Exceeded"]:
                summary["Total " + kw] = total_result[kw]
            else:
                summary["Avg. " + kw] = total_result[kw] / total_result_cnt[kw]
        
        if visualize:
            result_visualizer(summary, sys.stdout.write)
        return summary, write_results
    
    ## TODO generate adversarial samples
    