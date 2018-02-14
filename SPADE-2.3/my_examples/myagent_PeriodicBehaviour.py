import spade
import time

class MyAgent(spade.Agent.Agent): # default for all agents
	class MyBehav(spade.Behaviour.PeriodicBehaviour):
		def onStart(self):
			print("Starting behaviour . . .")
			self.counter = 0

		def _onTick(self): # sort of instead of process method (executed every period)
			print("Counter:"), self.counter
			self.counter = self.counter + 1

	def _setup(self):
		print("MyAgent starting . . .")
		b = self.MyBehav(1) # value 1 is the period of execution i [s]
		self.addBehaviour(b, None) # no process method! (internally by behaviour)

if __name__ == "__main__":
    
	a = MyAgent("agent@127.0.0.1", "secret")
	a.start()
