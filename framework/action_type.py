from config import consts


class ActionType:
    """
    Represents single environment action type.
    """

    def __init__(self, action_name, values_type, action_values):
        """
        Initiate action type instance.

        :param action_name: string. the name of the action.
        :param values_type: one of ['categorical', 'continues'].
        :param action_values: list.
            if values_type == 'continues', action_values = [min, max].
                any number between min and max is allowed.
            else:
                only values in action_values are allowed.
        """
        assert values_type in [
            consts.ActionTypeConsts.CATEGORICAL_VALUES_TYPE,
            consts.ActionTypeConsts.CONTINUES_VALUES_TYPE
        ], 'invalid values_type: {}'.format(values_type)

        self.__action_name = action_name
        self.__values_type = values_type
        self.__action_values = action_values

    @property
    def action_name(self):
        return self.__action_name

    @property
    def values_type(self):
        return self.__values_type

    @property
    def action_values(self):
        return self.__action_values

    def is_valid_value(self, action_value):
        """
        Check whether or not the given value is valid for the current action type.
        :param action_value: object.
        :return: boolean.
        """
        # handle categorical case
        if self.__values_type == consts.ActionTypeConsts.CATEGORICAL_VALUES_TYPE:
            return action_value in self.action_values

        else:   # continues case
            # extract range configuration
            min_value, max_value = self.action_values

            # verify value is in range
            return min_value <= action_value <= max_value
