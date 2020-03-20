
class face_idx(object):
    # face indexes
    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3
    TOP = 4
    BOTTOM = 5


class color_idx(object):
    # colors
    RED = 0
    BLUE = 1
    GREEN = 2
    YELLOW = 3
    CYAN = 4
    ORANGE = 5


class direction_idx(object):
    UP = 0
    DOWN = 1
    CCW = 2
    CW = 3


class moves_order(object):
    CCW_YAW = [face_idx.FRONT, face_idx.RIGHT, face_idx.BACK, face_idx.LEFT]
    UP_PITCH = [face_idx.FRONT, face_idx.TOP, face_idx.BACK, face_idx.BOTTOM]
    CCW_ROLL = [face_idx.TOP, face_idx.RIGHT, face_idx.BOTTOM, face_idx.LEFT]


class face_transformation_map(object):
    """
    How to transform the cube to yaw action.
    """
    ROLL = {
        face_idx.LEFT: [direction_idx.CCW],
        face_idx.BOTTOM: [direction_idx.CW, direction_idx.CW],
        face_idx.RIGHT: [direction_idx.CW],
    }

    PITCH = {
        face_idx.FRONT: [direction_idx.CCW],
        face_idx.TOP: [direction_idx.CCW],
        face_idx.BACK: [direction_idx.CW],
        face_idx.BOTTOM: [direction_idx.CCW],
    }