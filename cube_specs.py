
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


class face_transformation_map(object):
    PITCH = {
        (face_idx.FRONT, face_idx.TOP): 1,
        (face_idx.TOP, face_idx.BACK): -1,
        (face_idx.BACK, face_idx.BOTTOM): -1,
        (face_idx.BOTTOM, face_idx.FRONT): 1
    }

    @staticmethod
    def get_transformation(trans_map, face_pair):
        if face_pair in trans_map.keys():
            return trans_map[face_pair]
        elif face_pair[::-1] in trans_map.keys():
            return trans_map[face_pair]
        raise Exception('face pair not in map:', trans_map, 'pair:', face_pair)
