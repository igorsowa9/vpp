import time
from osbrain import run_agent
from osbrain import run_nameserver


def log_message(agent, message):
    agent.log_info('Received: %s' % message)


def log_message2(agent, message):
    agent.log_info('Received: %s' % message)



if __name__ == '__main__':

    # System deployment
    ns = run_nameserver()
    alice = run_agent('Alice')
    bob = run_agent('Bob')

    # System configuration
    addr = alice.bind('PUSH', alias='main')
    bob.connect(addr, handler=log_message)

    # Send messages
    for i in range(3):
        time.sleep(1)
        alice.send('main', 'Hello, Bob!')

    addr2 = alice.bind('PUSH', alias='main2')
    bob.connect(addr2, handler=log_message2)

    # Send messages
    for i in range(3):
        time.sleep(1)
        alice.send('main2', 'Hello, Bob2222222!')

    ns.shutdown()