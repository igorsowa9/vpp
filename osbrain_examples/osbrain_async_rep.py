import time

from osbrain import run_agent
from osbrain import run_nameserver


def reply_late(agent, message):
    time.sleep(1)
    return 'Hello, Bob!'


def process_reply(agent, message):
    agent.log_info('Processed reply: %s' % message)


def deaf(agent, message):
    agent.log_info('I am deaf...')


def no_reply_in_time(agent):
    agent.log_warning('No reply received!')


if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice')
    bob = run_agent('Bob')

    addr = alice.bind('ASYNC_REP', handler=reply_late)
    bob.connect(addr, alias='alice', handler=process_reply)

    bob.send('alice', 'Hello, Alice!', wait=0.5, on_error=no_reply_in_time)
    bob.log_info('I am done!')

    bob.log_info('Waiting for Alice to reply...')
    time.sleep(2)

    ns.shutdown()