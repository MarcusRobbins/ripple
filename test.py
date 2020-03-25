import random
import cv2
from deap import creator, base, tools, algorithms, cma
import rippleTank as rt
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import numpy as np
import time
from linetimer import CodeTimer
#from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D
import os 
from PIL import Image, ImageDraw
# import time
# import sys
# import functools
# from pvtrace import *
# import trimesh

tank_width = 600
tank_height = 600
numCells = 100.0
units = tank_width / numCells


globFitness = float('inf')
globSuccessRays = -1
prevSourceVal = 0
genomeSize = 140

img1_path = '/Users/bexus/Downloads/2_1.png'
img1 = cv2.imread(img1_path, 0)
img1_norm = img1 / 255.0

img2_path = '/Users/bexus/Downloads/2_2.png'
img2 = cv2.imread(img2_path, 0)
img2_norm = img2 / 255.0


def save_image(name, array):
    array = array*255.
    plt.imsave(name, array, cmap=matplotlib.cm.gray)#, vmin=-255, vmax=255)
    #plt.close(fig)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float)

numEvals = 0

def evalOneMax(individual):
    tank = rt.RippleTank(xdim=(-(tank_width/2), (tank_width/2)), ydim=(-(tank_height/2), (tank_height/2)), deep=100.0)
    x = 0.0
    y = 0.0
    freq = 1
    phase = 1
    amplitude = 1
    global numEvals
    numEvals += 1


    def createSineSource(data):
        def newFunc(source, i):
            answer = np.zeros_like(source.X_grid)
            r = int(round(i % genomeSize))            
            global prevSourceVal
            newVal = prevSourceVal + data[r]
            newVal = (1.0 * newVal) / (np.abs(newVal) + 0.5)
            #newVal = np.clip(newVal, -2.0, 2.0)
            answer[source.positions] = newVal
            prevSourceVal = newVal
            return answer
        return newFunc

    global prevSourceVal
    prevSourceVal = 0
    rt.Source(tank, createSineSource(individual), xcorners = (x-tank.dx - 300, x+tank.dx - 300), ycorners=(y-tank.dy, y+tank.dy), freq=freq, phase = phase, amplitude = amplitude)
    #rt.Source(tank, createSineSource(individual), xcorners = (x-tank.dx + 300, x+tank.dx + 300), ycorners=(y-tank.dy, y+tank.dy), freq=freq, phase = phase, amplitude = amplitude)

    data = tank.solvePoints(genomeSize)
    #with CodeTimer('create ripple'):
    #data = tank.solvePoints(2)

    # For each frame calculate the gradient
    fitness = 0
    successRays = 0
    start_index = genomeSize - 2
    num_items = genomeSize - start_index
    start_ray_test = 50

    for i in range(start_index, genomeSize-1):
    #for i in range(0, 1):
        frame = data[i]

        #x, y = np.meshgrid(range(frame.shape[0]), range(frame.shape[1]))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, frame)
        # plt.title('z as 3d height map')
        # plt.show()

        # show hight map in 2d
        # plt.figure()
        # plt.title('z as 2d heat map')
        # p = plt.imshow(frame)
        # plt.colorbar(p)
        # plt.show()

        gradients = np.gradient(frame)
        x_grad = gradients[1]
        y_grad = gradients[0]

        #x_grad[26][0] = -2

        #print("gradient: " + str(x_grad[26][0]))
        #print("arctan gradient: " + str(np.arctan(x_grad[26][0])))
        #print("sin arctan gradient: " + str(np.sin(np.arctan(x_grad[26][0]))))
        #print("arcsin refr arctan gradient: " + str(np.arcsin((1 / 1.333) * np.sin(np.arctan(x_grad[26][0])))))
        #print("90 degrees - arcsin refr arctan gradient: " + str((90.0 * np.pi / 180.0) - np.arcsin((1 / 1.333) * np.sin(np.arctan(x_grad[26][0])))))

        # x_reflected_gradients = np.tan((90.0 * np.pi / 180.0) - np.arcsin((1 / 1.333) * np.sin(np.arctan(x_grad)))) * -1
        # y_reflected_gradients = np.tan((90.0 * np.pi / 180.0) - np.arcsin((1 / 1.333) * np.sin(np.arctan(y_grad)))) * -1

        x_grad = np.diff(frame[49]) / units
        x_grad[3] = 3
        #x_grad = np.ones(99)
        x_reflected_gradients = np.tan(((90.0 * np.pi / 180.0) - np.arctan(x_grad)) + (np.arcsin((1.0/1.333) * np.sin(np.arctan(x_grad))))) * -1.0
        #x_reflected_gradients = np.tan(np.ones(99) * (87.0 * np.pi / 180.0))
        

        test = np.tan((90.0 * np.pi / 180.0) - (np.arcsin((1/1.333) * np.sin(np.arctan(3)))) + np.arctan(3)) * 1.0

        a = np.arctan(3)
        b = (1/1.333) * np.sin(np.arctan(3))
        c = (np.arcsin((1/1.333) * np.sin(np.arctan(3))))


        x_distance = (np.arange(100) + np.array(np.ones(100))[:, None] - 51.0) * 1.0
        y_distance = np.transpose(x_distance) * 1.0

        #x_reflected_gradients = individual[1:]

        z_from_x_distance = x_distance[49][1:] * x_reflected_gradients * units * -1.0
        #z_from_y_distance = y_distance * y_reflected_gradients
        z_from_y_distance = 0

        # OLD CODE
        # # Where the gradient is small the normal is going to shoot to infinity, try caping it:
        # x_grad = np.where((x_grad < 0.0001) & (x_grad >= 0), 0.0001, x_grad)
        # x_grad = np.where((x_grad > -0.0001) & (x_grad <= 0), -0.0001, x_grad)
        # y_grad = np.where((y_grad < 0.0001) & (y_grad >= 0), 0.0001, y_grad)
        # y_grad = np.where((y_grad > -0.0001) & (y_grad <= 0), -0.0001, y_grad)
        # z_from_x_distance = x_distance * (1.0 / x_grad) * -1.0
        # z_from_y_distance = y_distance * (1.0 / y_grad) * -1.0



        z_intercept = z_from_x_distance + z_from_y_distance    

        z_intercept = np.where(z_intercept > 0.0, 1.0, z_intercept)

        focal_point_z = -800

        z_intercept_targeted = z_intercept - focal_point_z

        #z_intercept_targeted = np.where(z_intercept_targeted > 0.0, 0.0, z_intercept_targeted)
        #z_intercept_targeted = np.where(z_intercept_targeted < -1000.0, -1000.0, z_intercept_targeted)
        z_intercept_targeted = (z_intercept_targeted) / (np.abs(z_intercept_targeted) + 0.5)

        fitness += np.sum(np.abs(np.square(z_intercept_targeted))) / (1.0 * float(num_items))

        target_size = 30.0
        z_intercept_point = np.where((z_intercept_targeted > -(target_size / 2.0)) & (z_intercept_targeted < (target_size / 2.0)), 1.0, 0.0)

        #if i > start_ray_test:
        successRays += np.sum(z_intercept_point) / (50.0 * 1.0)  #float(num_items - start_ray_test))
        
        #fitness += np.sum(z_intercept_point) / (50.0) #* float(num_items))

        

    #opp_over_adj =  np.divide(y_grad, x_grad)
    #angles = np.arctan(opp_over_adj)



    # For each gradient calculate the light hitting the collector
            
    #fitness = np.sum(np.absolute(magish))

    #diff1 = np.absolute(img1_norm - frame1)

    #fitness1 = abs(0 - np.sum(diff1))

    global globFitness
    global globSuccessRays
    if globFitness > fitness:
        globFitness = fitness
        #save_image('img_norm.png', img1_norm)
        #save_image('z_intercept.png', z_intercept)
        #save_image('z_intercept_targeted.png', 1 / z_intercept_targeted)        
        #save_image('z_intercept_point.png', z_intercept_point)
        #save_image('x_grad.png', x_grad)
        #save_image('y_grad.png', y_grad)
        save_image('frame.png', frame-1)

        #tank.simulateTime(10)
        #ani = tank.makeAnimation()
        #ani.save('simpleSource.gif', writer='imagemagick')



        # Draw image
        im = Image.new('RGBA', (6000, 6000), (0, 0, 0, 255)) 
        draw = ImageDraw.Draw(im) 
        z1 = 0.0
        scale = 10.0
        y_translate = 3800
        #x_grad = x_grad / 6
        #x_reflected_gradients = x_reflected_gradients / 6

        for i in range(1, 99):

            x1 = i * units * scale
            x2 = (i+1) * units * scale
            z2 = ((frame[49][i]) * scale) + y_translate

            draw.line((x1, z1, x2, z2), fill=(255, 255, 255))

            grad_z = x_grad[i-1] * scale * units
            draw.line((x1, z1, x2, z1 + (grad_z)), fill=(255, 255, 255))

            #x_middle = 60*(i + 0.5)
            #y_middle = (prevZ + grad_z)/2

            ray_length = 45

            if x_reflected_gradients[i-1] < 10000 and x_reflected_gradients[i-1] > -10000: # and i == 49:

                grad_z_refl = x_reflected_gradients[i-1] * (scale * units)

                # init_x = ((60*i) + (60*(i+1)))/2.0
                # init_y = (prevZ + prevZ + grad_z) / 2.0

                x1 = x1 + 20

                #draw.line((x1, z1, x2, z1 + grad_z_refl), fill=(0, 255, 0))

                draw.line((x1, z1, x1 - (units * scale)*ray_length, (z1 - grad_z_refl*ray_length)), fill=(0, 255, 0))
                draw.line((x1, z1, x1 + (units * scale)*ray_length, (z1 + grad_z_refl*ray_length)), fill=(0, 255, 0))

            z1 = z2

        # Draw collector
        draw.ellipse(((units * scale) * 50.5 - 10, focal_point_z * scale + y_translate + 1000 - (target_size * scale / 2.0), (units * scale) * 50.5 + 10, focal_point_z * scale + y_translate + 1000 + (target_size * scale / 2.0)), fill=(255, 255, 0))

        # draw the z intercepts
        for t in z_intercept * scale * 1.0 + y_translate + 1000:
            draw.ellipse(((units * scale) * 50.5 - 10,(t) - 10, (units * scale) * 50.5 + 10,(t) + 10), fill=128)

        draw.line((0, y_translate, (units * scale * numCells), y_translate), fill=128)

        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        #im.show()
        im.save("./out/test" + str(numEvals) + ".png")



        # Lets dump out a file to view in ray tracer!!!!
        it = map(lambda x: "{\"x\":" + str(x[0] * units) + ",\"y\":" + str(x[1]) + ",\"arc\":false},", enumerate(frame[49]))
        coords = ''.join(it)[:-1]

        first = "{\"version\":2,\"objs\":[{\"type\":\"parallel\",\"p2\":{\"type\":1,\"x\":600,\"y\":50,\"exist\":true},\"p1\":{\"type\":1,\"x\":0,\"y\":50,\"exist\":true},\"p\":0.5},{\"type\":\"refractor\",\"path\":[" + coords
        last = ",{\"y\":-800,\"x\":" + str(genomeSize * units) + ",\"arc\":false},{\"y\":-800,\"x\":" + str(0) + ",\"arc\":false}],\"notDone\":false,\"p\":1.5}],\"mode\":\"light\",\"rayDensity_light\":0.1,\"rayDensity_images\":1,\"observer\":null,\"origin\":{\"x\":542.9644690819766,\"y\":70.20970720433743},\"scale\":0.7289999999999995}"

        allt = first + last

        text_file = open("Output.txt", "w")
        text_file.write(allt)
        text_file.close()






    if globSuccessRays < successRays:
        globSuccessRays = successRays
        print("Success rays: " + str(successRays))

    return fitness,


toolbox.register("evaluate", evalOneMax)

np.random.seed(128)
N=genomeSize
strategy = cma.Strategy(centroid=[0.0]*N, sigma=0.1, lambda_=6)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaGenerateUpdate(toolbox, ngen=10000000, stats=stats, halloffame=hof)
