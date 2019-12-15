
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import*
import matplotlib.pyplot as plt


class CubePlotter(object):
    """
    Plots a cube.
    """
    COLOR_MAP = [
        'red',
        'blue',
        'green',
        'yellow',
        'cyan',
        'orange'
    ]

    AX_SHIFT = 0.5

    @staticmethod
    def get_face_cords(face_idx):

        if face_idx == 0:   # front

            xs = np.zeros([3, 3])
            xs[:, 1] = 0.5
            xs[:, 2] = 1.0

            ys = np.ones([3, 3]) + CubePlotter.AX_SHIFT

            zs = np.zeros([3, 3])
            zs[0, :] = 1.0
            zs[1, :] = 0.5

        elif face_idx == 1:   # right

            xs = np.ones([3, 3]) + CubePlotter.AX_SHIFT

            ys = np.zeros([3, 3])
            ys[:, 0] = 1.0
            ys[:, 1] = 0.5

            zs = np.zeros([3, 3])
            zs[0, :] = 1.0
            zs[1, :] = 0.5

        elif face_idx == 2:   # back

            xs = np.zeros([3, 3])
            xs[:, 0] = 1.0
            xs[:, 1] = 0.5

            ys = np.zeros([3, 3]) - CubePlotter.AX_SHIFT

            zs = np.zeros([3, 3])
            zs[0, :] = 1.0
            zs[1, :] = 0.5

        elif face_idx == 3:   # left

            xs = np.zeros([3, 3]) - CubePlotter.AX_SHIFT

            ys = np.zeros([3, 3])
            ys[:, 1] = 0.5
            ys[:, 2] = 1.0

            zs = np.zeros([3, 3])
            zs[0, :] = 1.0
            zs[1, :] = 0.5

        elif face_idx == 4:   # top

            xs = np.zeros([3, 3])
            xs[:, 1] = 0.5
            xs[:, 2] = 1.0

            ys = np.zeros([3, 3])
            ys[1, :] = 0.5
            ys[2, :] = 1.0

            zs = np.ones([3, 3]) + CubePlotter.AX_SHIFT

        else:                   # face_idx == 5:   bottom

            xs = np.zeros([3, 3])
            xs[:, 1] = 0.5
            xs[:, 2] = 1.0

            ys = np.zeros([3, 3])
            ys[0, :] = 1.0
            ys[1, :] = 0.5

            zs = np.zeros([3, 3]) - CubePlotter.AX_SHIFT

        return xs, ys, zs

    @staticmethod
    def plot_face(face_colors, ax, xs, ys, zs):

        colors = list(map(lambda color_idx: CubePlotter.COLOR_MAP[color_idx], face_colors.reshape(-1)))

        ax.scatter(xs.reshape(-1), ys.reshape(-1), zs.reshape(-1),
                   marker='o', s=2500, c=colors, alpha=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    @staticmethod
    def plot_3d_cube(cube_arr):
        """
                    Z
                   /|\
                    |
                    |   /
                    | /
             -------|------------> X
                  / |
                /   |
              /     |
           |/_
          Y

        Faces:
        0 - Front   - y = 1
        1 - Right   - x = 1
        2 - Back    - y = 0
        3 - Left    - x = 0
        4 - Top     - z = 1
        5 - Bottom  - z = 0

        :param cube_arr: np.ndarray.
                        shape: [6, 9].
                        face order: cube_specs.face_idx values
        :return:
        """

        # create figure and axe
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot front dots
        for face_idx in range(6):
            xs, ys, zs = CubePlotter.get_face_cords(face_idx)
            CubePlotter.plot_face(cube_arr[face_idx], ax, xs, ys, zs)

        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    test_cube = np.zeros(shape=[6, 3, 3]).astype(int)

    test_cube[3, :, 2] = 1
    test_cube[0, :, 0] = 1

    CubePlotter.plot_3d_cube(test_cube)
