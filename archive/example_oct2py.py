from osbrain import Agent
from oct2py import octave
import time, sys
from osbrain import run_agent
from osbrain import run_nameserver

octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')

mpc = octave.case5()
res = octave.runopf(mpc, octave.mpoption('out.all', 1))


class MyAgent1(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')

    def hello(self, name):
        self.send('main', 'Hello, %s!' % name)

    def custom_log(self, message):
        self.log_info('Received: %s' % message)

    def octave_test(self):
        mpc = octave.case5()
        res = octave.runopf(mpc, octave.mpoption('out.all', 1))
        self.set_attr(result=res)
        return res

    def pypower_test(self):
        ppc = cases[data['case']]()
        r = rundcopf(ppc)
        return r['success']

    def sent_opf_result(self):
        pass


if __name__ == '__main__':

    # System deployment
    ns = run_nameserver()
    alice = run_agent('Alice', base=MyAgent1)
    bob = run_agent('Bob', base=MyAgent1)

    # System configuration
    bob.connect(alice.addr('main'), handler='custom_log')

    # Send messages
    for _ in range(3):
        alice.hello('Bob')
        time.sleep(1)

    ns.shutdown()
