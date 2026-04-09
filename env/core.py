from typing import Tuple, Dict, Any
from env.models import Observation, Action, Reward
<<<<<<< HEAD
from env.reward_policy import finalize_open_score
=======
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
from env.state import EnvState


class MisinfoEnv:
    """
    Main OpenEnv Environment
    """

    def __init__(self, data, grader):
        self.state = EnvState(data)
        self.grader = grader

    def reset(self) -> Observation:
        """
        Reset environment and return first observation
        """
        self.state.reset()
        sample = self.state.current_sample()

        return Observation(
            claim=sample.get("claim", ""),
            context=sample.get("context", ""),
            history=[]
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply action → return next observation, reward, done, info
        """
        sample = self.state.current_sample()

        # ✅ Grade the action
        reward_obj: Reward = self.grader.grade(sample, action)
<<<<<<< HEAD
        safe_reward = finalize_open_score(reward_obj.score)
=======
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

        # ✅ Store history
        self.state.add_history(
            f"Claim: {sample.get('claim')} | Action: {action.label} ({action.confidence})"
        )

        # ✅ Move forward
        self.state.advance()

        done = self.state.is_done()

        # ✅ Next observation
        next_sample = self.state.current_sample()

        observation = Observation(
            claim=next_sample.get("claim", "") if not done else "",
            context=next_sample.get("context", "") if not done else "",
            history=self.state.history
        )

<<<<<<< HEAD
        return observation, safe_reward, done, {
=======
        return observation, reward_obj.score, done, {
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
            "feedback": reward_obj.feedback
        }

    def state_info(self) -> Dict[str, Any]:
        """
        Return current state (for debugging / evaluation)
        """
        return {
            "current_index": self.state.current_index,
            "done": self.state.done,
            "history": self.state.history
<<<<<<< HEAD
        }
=======
        }
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
