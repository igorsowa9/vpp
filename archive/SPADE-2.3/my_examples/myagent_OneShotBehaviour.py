import spade
import time

class MyAgent(spade.Agent.Agent): # default for all agents
	class MyBehav(spade.Behaviour.OneShotBehaviour):
		def onStart(self):
			print "Starting behaviour . . ."

		def _process(self):
                        print "Hello world from OneShot"

                def onEnd(self):
                        print "Ending Behaviour"

	def _setup(self):
		print "MyAgent starting . . ."
		b = self.MyBehav()
		self.addBehaviour(b, None)

if __name__ == "__main__":
	a = MyAgent("agent@127.0.0.1", "secret")
	a.start()
