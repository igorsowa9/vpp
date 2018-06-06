from osbrain import run_nameserver
from osbrain import run_agent

if __name__ == '__main__':

    # System deployment
    ns = run_nameserver()
    run_agent('Agent0')
    run_agent('Agent1')
    run_agent('Agent2')

    # Show agents registered in the name server
    agent = ns.proxy('Agent1')
    agent.log_info('Hello world!')

    ns.shutdown()