from osbrain import run_agent
from osbrain import run_nameserver
import time


def reply_func(agent, message):
    yield "sth for bob"
    time.sleep(.1)
    agent.log_info(message)


if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice')
    john = run_agent('John')
    bob = run_agent('Bob')

    addr = alice.bind('REP', handler=reply_func)
    addr2 = john.bind('REP', handler=reply_func)

    bob.connect(addr, alias='main')
    bob.connect(addr2, alias='main2')

    for i in range(10):
        bob.send('main', i)
        bob.send('main2', i+.5)
        print(bob.recv('main'))
        print(bob.recv('main2'))

    ns.shutdown()
