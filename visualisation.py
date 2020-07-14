import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import HTML
import imageio
from GameLogic import Game, Point


class Helpers:
    def __init__(self,  env: Game):
        self.env = env

    def show_hist_map(self, hist):
        for i in range(len(hist)):
            if all(elm < self.env.board_size[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)

                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.env.board_size[0], self.env.board_size[1]), cmap='hot', interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.env.board_size[0], self.env.board_size[1])))
                plt.show()

    def show_reward(self, h, elem_max=True):
        m = np.zeros(self.env.board_size)
        for i in range(self.env.board_size[0]):
            for j in range(self.env.board_size[1]):
                reward, _ = self.env.get_reward_for_position(Point(i, j))
                if reward > 30 and elem_max:
                    m[i, j] = 100
                else:
                    m[i, j] = reward

        return m.astype(np.int)

    def fig_to_data(self, fig):
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()

        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)

        buf = np.roll(buf, 3, axis=2)
        plt.close()
        return buf

    def show_hist_ani(self, hist):
        frame_array = []
        for i in range(len(hist)):
            if all(elm < self.env.board_size[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)
                fig = plt.figure()
                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.env.board_size[0], self.env.board_size[1]), cmap='hot', interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.env.board_size[0], self.env.board_size[1])))
                frame_array.append(self.fig_to_data(fig))
        #

        w = imageio.get_writer('output.mp4', fps=6, quality=6)
        for i in range(len(frame_array)):
            w.append_data(frame_array[i])
        w.close()

    # def playVideo(self, path):
    #     before = """<video width="864" height="576" controls><source src="""
    #     end = """ type="video/mp4"></video>"""
    #     return HTML(before + path + end)