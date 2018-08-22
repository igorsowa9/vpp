import spade
import time

class MyAgent(spade.Agent.Agent): # default for all agents
	class MyBehav(spade.Behaviour.TimeOutBehaviour):
		def onStart(self):
			print "Starting behaviour . . ."

		def timeOut(self):
                        print "The timeout has ended"

                def onEnd(self):
                        print "Ending Behaviour . . ."

	def _setup(self):
		print "MyAgent starting . . ."
		b = self.MyBehav(5) # timeout value (delay before behaviour)
		self.addBehaviour(b, None)

if __name__ == "__main__":
	a = MyAgent("agent@127.0.0.1", "secret")
	a.start()
