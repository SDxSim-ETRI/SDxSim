from BaseEnvironment import BaseEnvironment

class CustomEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.state = None
        self.done = False

    def reset(self):
        self.state = [0, 0, 0]  # Example initial state
        self.done = False
        return self.state

    def isterminal(self):
        return self.done

    def step(self, action):
        # Example step logic
        self.state = [s + a for s, a in zip(self.state, action)]
        reward = -sum(self.state)  # Example reward
        self.done = sum(self.state) > 10  # Example terminal condition
        return self.state, reward, self.done

    def setaction(self, action):
        self.action = action

    def getreward(self):
        return -sum(self.state)

    def getstate(self):
        return self.state
