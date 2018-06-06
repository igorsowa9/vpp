from osbrain import run_agent
from osbrain import run_nameserver


def reply(agent, message):
    return 'Received ' + str(message)

def reply2(agent, message):
    agent.log_info('Doing some stuff before sending back!')   # Do some stuff first
    yield 'Received ' + str(message)  # Reply now

if __name__ == '__main__':

    ns = run_nameserver()
    alice = run_agent('Alice')
    bob = run_agent('Bob')

    addr1 = alice.bind('REP', handler=reply2)
    addr2 = alice.bind('REP', handler=reply2)

    bob.connect(addr1, alias='main1')
    bob.connect(addr2, alias='main2')

    bob.send('main1', 5)
    bob.send('main2', 3.14)

    reply1 = bob.recv('main1')
    reply2 = bob.recv('main2')

    print(reply1 + ' received by bob.')
    print(reply2 + ' received by bob.')

    ns.shutdown()