from typing import List, Dict


class EnvState:
    """
    Maintains the internal state of the environment
    """

    def __init__(self, data: List[Dict]):
        self.data = data
        self.current_index = 0
        self.done = False
        self.history = []  # stores past actions or logs

    def reset(self):
        """
        Reset environment state
        """
        self.current_index = 0
        self.done = False
        self.history = []

    def current_sample(self) -> Dict:
        """
        Get current data sample
        """
        if self.current_index < len(self.data):
            return self.data[self.current_index]
        return {}

    def advance(self):
        """
        Move to next step
        """
        self.current_index += 1

        if self.current_index >= len(self.data):
            self.done = True

    def add_history(self, entry: str):
        """
        Store action/decision history
        """
        self.history.append(entry)

    def is_done(self) -> bool:
        """
        Check if episode is finished
        """
        return self.done