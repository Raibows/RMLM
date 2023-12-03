from ..victim import Victim
from ..exceptions import InvokeLimitExceeded, AttackException
import logging

logger = logging.getLogger("OpenAttack.AttackEval")

def attack_process(attacker, victim : Victim, data, limit):
    victim.set_context(data, limit)
    OOM_FAIL = False
    prob = None
    try:
        adversarial_sample, prob = attacker(victim, data)
        invoke_times = victim.context.invoke
        attack_time = victim.context.attack_time
    except InvokeLimitExceeded:
        adversarial_sample = None
        invoke_times = victim.context.invoke + 1
        attack_time = victim.context.attack_time
    except KeyboardInterrupt as e:
        raise e
    except AttackException:
        adversarial_sample = None
        invoke_times = victim.context.invoke
        attack_time = victim.context.attack_time
    except Exception as e:
        logger.exception(f"Exception fail with\n{e}\n")
        adversarial_sample = None
        invoke_times = victim.context.invoke
        attack_time = victim.context.attack_time
        OOM_FAIL = True
    finally:
        victim.clear_context()
    
    return adversarial_sample, attack_time, invoke_times, OOM_FAIL, prob
    

def worker_process(data):
    attacker = globals()["$WORKER_ATTACKER"]
    victim = globals()["$WORKER_VICTIM"]
    limit = globals()["$WORKER_INVOKE_LIMIT"]

    return attack_process(attacker, victim, data, limit)



def worker_init(attacker, victim, limit):
    globals()['$WORKER_ATTACKER'] = attacker
    globals()['$WORKER_VICTIM'] = victim
    globals()['$WORKER_INVOKE_LIMIT'] = limit
