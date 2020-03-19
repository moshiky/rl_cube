
class State:
    """
    Represents single environment state.
    """

    def __init__(self, features, is_final):
        """
        Init environment state.

        :param features: object. state features. can be of any serializable format.
        :param is_final: boolean. whether or not the represented state is final state.
        """
        self.__features = features
        self.__is_final = is_final

    @property
    def features(self):
        return self.__features

    @property
    def is_final(self):
        return self.__is_final
