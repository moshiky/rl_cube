
import numpy as np

from simulators.rubiks_cube import cube_specs, utils


class RubiksCube(object):
    """
    Represents rubik's cube and all necessary methods.
    """

    def __init__(self, edge_size):
        """

        :param edge_size:
        """
        self._edge_size = edge_size
        self._cube_state = np.zeros(shape=[6, edge_size, edge_size])
        for face_idx in range(self._cube_state.shape[0]):
            self._cube_state[face_idx] = face_idx

    def get_cube(self):
        return self._cube_state.copy()

    def yaw(self, row_idx, direction):
        """
        Rotation of horizontal row of Front face.

        :param row_idx: [0 - edge_size-1]
        :param direction: [cw or ccw]
        :return:
        """
        # validate inputs
        assert 0 <= row_idx < self._edge_size, 'Invalid row idx!'
        assert direction in [cube_specs.direction_idx.CCW, cube_specs.direction_idx.CW], 'Invalid direction!'

        # set moves order
        if direction == cube_specs.direction_idx.CCW:
            move_order = cube_specs.moves_order.CCW_YAW
        else:
            move_order = cube_specs.moves_order.CCW_YAW[::-1]

        rotation_faces = [cube_specs.face_idx.TOP, cube_specs.face_idx.BOTTOM]
        rotation_face_directions = [cube_specs.direction_idx.CCW, cube_specs.direction_idx.CW]
        self._yaw(row_idx, direction, move_order, rotation_faces, rotation_face_directions)

    def _yaw(self, row_idx, direction, move_order, rotation_faces, rotation_face_directions):
        """
        Rotation of horizontal row of Front face.

        :param row_idx: [0 - edge_size-1]
        :param direction: [cw or ccw]
        :param rotation_faces: list:
            [0] = face_idx to rotate in case row_id = 0
            [1] = face_idx to rotate in case row_id != 0
        :param rotation_face_directions: given ccw, how to rotate each face.
        :return:
        """
        # perform yaw movement on the row
        previous_face_row = self._cube_state[move_order[0], row_idx, :].copy()
        for move_idx in range(len(move_order)):
            target_face_idx = move_order[(move_idx + 1) % len(move_order)]
            current_face_row = self._cube_state[target_face_idx, row_idx, :].copy()
            self._cube_state[target_face_idx, row_idx, :] = previous_face_row
            previous_face_row = current_face_row

        # rotate relevant face if rotating upper or lower rows
        if row_idx in [0, self._edge_size - 1]:
            face_rotation_direction = rotation_face_directions[0] if row_idx == 0 else rotation_face_directions[1]
            if direction == cube_specs.direction_idx.CW:
                face_rotation_direction = utils.swap_direction(face_rotation_direction)

            face_idx = rotation_faces[0] if row_idx == 0 else rotation_faces[1]
            self.rotate_face(face_idx, face_rotation_direction)

    def rotate_face(self, face_idx, face_rotation_direction):
        """
        Rotates a face in the specified direction.

        :param face_idx:
        :param face_rotation_direction: [cw or ccw]
        :return:
        """
        # validate rotation direction
        assert face_rotation_direction in [cube_specs.direction_idx.CCW, cube_specs.direction_idx.CW], \
            'Invalid face rotation direction!'

        # rotate face
        if face_rotation_direction == cube_specs.direction_idx.CW:
            self._cube_state[face_idx] = self._cube_state[face_idx].T[:, ::-1]
        else:
            self._cube_state[face_idx] = self._cube_state[face_idx].T[::-1, :]

    def pitch(self, column_idx, direction):
        """
        Rotation of vertical column of Front face.

        :param column_idx: [0 - edge_size-1]
        :param direction: [up or down]
        :return:
        """
        # validate inputs
        assert 0 <= column_idx < self._edge_size, 'Invalid column idx!'
        assert direction in [cube_specs.direction_idx.UP, cube_specs.direction_idx.DOWN], 'Invalid direction!'

        # set moves order
        if direction == cube_specs.direction_idx.UP:
            move_order = cube_specs.moves_order.UP_PITCH
        else:
            move_order = cube_specs.moves_order.UP_PITCH[::-1]

        # convert the move to yaw
        for face_idx, rotations in cube_specs.face_transformation_map.PITCH.items():
            for rotation_direction in rotations:
                self.rotate_face(face_idx, rotation_direction)

        # translate direction from UP/DOWN to CW/CCW
        if direction == cube_specs.direction_idx.UP:
            direction = cube_specs.direction_idx.CW
        else:
            direction = cube_specs.direction_idx.CCW

        # do yaw action
        rotation_faces = [cube_specs.face_idx.LEFT, cube_specs.face_idx.RIGHT]
        rotation_face_directions = [cube_specs.direction_idx.CW, cube_specs.direction_idx.CCW]
        self._yaw(column_idx, direction, move_order, rotation_faces, rotation_face_directions)

        # rotate the cube back to normal mode
        for face_idx, rotations in cube_specs.face_transformation_map.PITCH.items():
            for rotation_direction in rotations:
                self.rotate_face(face_idx, utils.swap_direction(rotation_direction))

    def roll(self, row_idx, direction):
        """
        Rotation of horizontal row of Top face.

        :param row_idx: [0 - edge_size-1]
        :param direction: [cw or ccw]
        :return:
        """
        # validate inputs
        assert 0 <= row_idx < self._edge_size, 'Invalid row idx!'
        assert direction in [cube_specs.direction_idx.CW, cube_specs.direction_idx.CCW], 'Invalid direction!'

        # set moves order
        if direction == cube_specs.direction_idx.CCW:
            move_order = cube_specs.moves_order.CCW_ROLL
        else:
            move_order = cube_specs.moves_order.CCW_ROLL[::-1]

        # convert the move to yaw
        for face_idx, rotations in cube_specs.face_transformation_map.ROLL.items():
            for rotation_direction in rotations:
                self.rotate_face(face_idx, rotation_direction)

        # do yaw action
        rotation_faces = [cube_specs.face_idx.BACK, cube_specs.face_idx.FRONT]
        rotation_face_directions = [cube_specs.direction_idx.CCW, cube_specs.direction_idx.CCW]
        self._yaw(row_idx, direction, move_order, rotation_faces, rotation_face_directions)

        # rotate the cube back to normal mode
        for face_idx, rotations in cube_specs.face_transformation_map.ROLL.items():
            for rotation_direction in rotations:
                self.rotate_face(face_idx, utils.swap_direction(rotation_direction))

    def scramble(self, num_moves):
        """
        Perform random move set to scramble the cube.

        :param num_moves: integer. number of moves to execute.
        :return: selected moves information.
        """
        # define scrambling moves order
        direction_map = {
            self.roll: [cube_specs.direction_idx.CW, cube_specs.direction_idx.CCW],
            self.pitch: [cube_specs.direction_idx.UP, cube_specs.direction_idx.DOWN],
            self.yaw: [cube_specs.direction_idx.CW, cube_specs.direction_idx.CCW],
        }

        # randomize moves
        selected_moves = np.random.choice(list(direction_map.keys()), size=num_moves)
        scrambling_moves = [[
            move_func,
            np.random.randint(self._edge_size),
            np.random.choice(direction_map[move_func])
        ] for move_func in selected_moves]

        # execute scrambling moves
        for move_func, idx, move_direction in scrambling_moves:
            move_func(idx, move_direction)

        return scrambling_moves

    def is_solved(self):
        """
        Check whether or not the cube is solved.

        :return: boolean.
        """
        # define colors bank
        colors_bank = list(range(self._cube_state.shape[0]))

        # search for matches
        for face_arr in self._cube_state:
            # get face colors
            face_colors = list(set(face_arr.reshape(-1).tolist()))

            # validate single value
            if len(face_colors) != 1:
                return False

            # validate color not selected yet
            if face_colors[0] not in colors_bank:
                return False

            # face is ok- remove color from colors bank
            colors_bank.remove(face_colors[0])

        # if got here- the cube is solved
        return True
