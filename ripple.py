import rippleTank as rt
import numpy as np
import matplotlib.pyplot as plt
import cv2

# creates a ripple tank
tank = rt.RippleTank()

#randF = np.random.normal(size=10)
randF = np.ones(10)

def sineSource(source, i):
    """
    Sine function.

    Returns:
        np.ndarray: array with sine values on source positions.
    """
    t = source.rippletank.dt*i
    answer = np.zeros_like(source.X_grid)
    #value = np.sin(2*np.pi*source.freq*t + source.phase)
    # if value < 0:
    #     answer[source.positions] = value
    # else:
    #     # return None
    r = int(round(i % 10))
    answer[source.positions] = randF[r]
    #print(randF[r])
    return answer


# creates a source on the ripple tank
rt.Source(tank, rt.dropSource, freq=10)

tank.simulateTime(10)
frame = tank.captureFrame()
#data = tank.solvePoints(100)

#frame = data[99]


ani = tank.makeAnimation()
# ani.save('simpleSource.gif', writer='imagemagick')

plt.show()
