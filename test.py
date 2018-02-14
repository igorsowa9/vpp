from osbrain import run_agent
from osbrain import run_nameserver


def reply_of_request(agent, message):
    return 'Received ' + str(message) + ' from ' + agent.name


if __name__ == '__main__':

    run_nameserver()
    alice = run_agent('Alice')
    bob = run_agent('Bob')
    eve = run_agent('Eve')

    alice.bind('REP', 'rrAlice', handler=reply_of_request)
    bob.connect(alice.addr('rrAlice'), alias='BobToAlice')

    eve.bind('REP', 'rrEve', handler=reply_of_request)
    bob.connect(eve.addr('rrEve'), alias='BobToEve')

    for i in range(10):
        bob.send('BobToAlice', i)
        bob.send('BobToEve', i+0.5)
        reply_to_bob1 = bob.recv('BobToAlice')
        reply_to_bob2 = bob.recv('BobToEve')
        print(reply_to_bob1)
        print(reply_to_bob2)
