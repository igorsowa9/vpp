from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent


def log_message(self, message):
    self.log_info('Received: %s' % message)


def handler2(self, message):
    self.log_info('Handler2: %s' % message)


class Greeter(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello, %s!' % name)


if __name__ == '__main__':

    # System deployment
    ns = run_nameserver()
    alice = run_agent('Alice', base=Greeter)
    bob = run_agent('Bob')
    run_agent('Agent2')

    # show all agents
    for alias in ns.agents():
        print(alias)

    # push pull a-b
    addr = alice.bind('PUSH', alias='main')
    bob.connect(addr, handler=[log_message, handler2]) #  This handler will be serialized and stored in the remote agent to be executed there when needed.

    ns.shutdown()