
import numpy as np
import matplotlib.pyplot as plt
from GameLogic import game, obstacle
from IPython.display import HTML
import imageio

class helpers:
    def __init__(self,  obstacles, env,boardSize=(10, 10), MAX_REWARD=100, safeDist=3  ):

        #self.aim = aim(aimPos[0], aimPos[1])
        self.boardSize = boardSize
        self.obstacles = obstacles
        self.boardSize = boardSize
        self.MAX_REWARD = MAX_REWARD
        self.safeDist = safeDist
        self.env = env

    def showhist_map(self, hist):
        for i in range(len(hist)):
            if all(elm < self.boardSize[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)

                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.boardSize[0], self.boardSize[1]), cmap='hot', interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.boardSize[0], self.boardSize[1])))
                plt.show()

    def showReward(self,h, elemMax=True):
        m = np.zeros(self.boardSize)
        for i in range(self.boardSize[0]):
            for j in range(self.boardSize[1]):
                m[i, j], _ = self.env.getRewardForField(i, j)
                if (m[i, j] > 30):
                    if elemMax:
                        m[i, j] = 100

        return m.astype(np.int)

    def fig2data(self,fig):
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()

        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)

        buf = np.roll(buf, 3, axis=2)
        plt.close()
        return buf

    def showhist_ani(self,hist):
        frame_array = []
        for i in range(len(hist)):
            if all(elm < self.boardSize[0] for elm in hist[int(i)]):
                h = hist[i].astype(np.int)
                fig = plt.figure()
                if np.max(hist[i]) == 1:
                    plt.imshow(hist[i].reshape(self.boardSize[0], self.boardSize[1]), cmap='hot', interpolation='nearest')
                else:
                    plt.imshow(np.zeros((self.boardSize[0], self.boardSize[1])))
                frame_array.append(self.fig2data(fig))
        #

        w = imageio.get_writer('output.mp4', fps=6, quality=6)
        for i in range(len(frame_array)):
            w.append_data(frame_array[i])
        w.close()

    def playVideo(self,path):
        before = """<video width="864" height="576" controls><source src="""
        end = """ type="video/mp4"></video>"""
        return HTML(before + path + end)