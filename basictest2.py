import sys
from osbrain import run_agent
from osbrain import run_nameserver


def doSomething1(param1):
    for alias in ns.agents():
        agent = ns.proxy(alias)
        print("Agent: " + alias + "Status: " + str(agent.get_attr('s')))


def doSomething2(param2):
    for alias in ns.agents():
        agent = ns.proxy(alias)
        print("Second task Value: " + str(agent.get_attr('v')))


if __name__ == '__main__':

    ns = run_nameserver()
    run_agent('Alice', attributes=dict(s='E', v=2))
    run_agent('Bob', attributes=dict(s='D', v=-3))
    run_agent('Charlie', attributes=dict(s='B', v=0))


    doSomething1(1)
    doSomething2(3)

