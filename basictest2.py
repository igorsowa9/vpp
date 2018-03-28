import time

from osbrain import run_agent
from osbrain import run_nameserver


def reply_late(agent, message):
    time.sleep(1)
    agent.log_info('I received: ' + message)
    return 'Hello, Bob!'


def process_reply(agent, message):
    agent.log_info('Processed reply: %s' % message)


def deaf(agent, message):
    agent.log_info('I am deaf...')


if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice')
    bob = run_agent('Bob')

    addr = alice.bind('ASYNC_REP', handler=reply_late)
    bob.connect(addr, alias='main', handler=process_reply)

    #bob.send('main', 'Hello, Alice!')
    bob.send('main', 'Hello, Alice!', handler=deaf)
    bob.log_info('I am done!')

    bob.log_info('Waiting for Alice to reply...')
    time.sleep(2)

    ns.shutdown()