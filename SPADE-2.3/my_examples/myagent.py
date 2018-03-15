import spade

class MyAgent(spade.Agent.Agent):
	class ReceiveBehav(spade.Behaviour.Behaviour):
		"""This behaviour will receive all kind of messages"""

		def _process(self):
			self.msg = None
			
			# Blocking receive for 10 seconds
			self.msg = self._receive(True, 10)
			
			# Check wether the message arrived
			if self.msg:
				print("I got a message!")
			else:
				print("I waited but got no message")

	class AnotherBehav(spade.Behaviour.Behaviour):
		"""This behaviour will receive only messages of the 'cooking' ontology"""
	
		def _process(self):
			self.msg = None
			
			# Blocking receive indefinitely
			self.msg = self._receive(True)
			
			# Check wether the message arrived
			if self.msg:
				print("I got a cooking message!")
			else:
				print("I waited but got no cooking message")				

	def _setup(self):
		# Add the "ReceiveBehav" as the default behaviour
		rb = self.ReceiveBehav()
		self.setDefaultBehaviour(rb)
		
		# Prepare template for "AnotherBehav"
		cooking_template = spade.Behaviour.ACLTemplate()
		cooking_template.setOntology("cooking")
		mt = spade.Behaviour.MessageTemplate(cooking_template)

		# Add the behaviour WITH the template
		self.addBehaviour(rb, mt)				

if __name__ == "__main__":
	a = MyAgent("agent@127.0.0.1", "secret")
	a.start()
