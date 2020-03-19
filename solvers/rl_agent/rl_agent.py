
class RLAgent(object):
    """
    This class represents an RL agent.
    Uses an IAgentLogic instance to learn.
    """
    def __init__(self, agent_logic):
        """
        Initiate agent instance.

        :param agent_logic: IAgentLogic instance.
        """
        self._agent_logic = agent_logic