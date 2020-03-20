
class Action:
    """
    Represents single environment action.
    """

    def __init__(self, action_value, action_cls):
        """
        Initiate action instance.

        :param action_value: object. selected value.
        :param action_cls: instance of ActionType.
        """
        # verify the action value is valid
        assert action_cls.is_valid_value(action_value), 'invalid action value: type= {}, value= {}'.format(
            action_cls.action_name, action_value
        )

        # store to internal members
        self.__action_value = action_value
        self.__action_cls = action_cls

    @property
    def action_value(self):
        return self.__action_value
