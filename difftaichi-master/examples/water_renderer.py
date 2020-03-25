import taichi as ti
import math
import numpy as np
import os
from cv2 import cv2

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda)

n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
max_steps = 256
vis_interval = 32
output_vis_interval = 1
steps = 256
amplify = 2

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(2, dt=real)

p = scalar()
rendered = scalar()
target = scalar()
initial = scalar()
loss = scalar()
height_gradient = vec()

wave_gen = scalar()

bottom_image = scalar()
refracted_image = scalar()
target_img = scalar()

mode = 'refract'

num_generators = 10

# def point(h, k, r):
#     theta = random() * 2 * pi
#     return h + cos(theta) * r, k + sin(theta) * r

# xy_circle = [point(50,50,100) for _ in range(num_generators)]

mask = scalar()

@ti.layout
def place():
  ti.root.dense(ti.l, max_steps).dense(ti.ij, n_grid).place(p)
  ti.root.dense(ti.ij, n_grid).place(rendered)
  ti.root.dense(ti.ij, n_grid).place(target)
  ti.root.dense(ti.ij, n_grid).place(initial)
  ti.root.dense(ti.ij, n_grid).place(height_gradient)
  ti.root.dense(ti.l, max_steps).dense(ti.i, n_grid).place(wave_gen)
  ti.root.dense(ti.ij, (n_grid, n_grid)).place(mask)
  ti.root.dense(ti.ijk, (n_grid, n_grid, 3)).place(bottom_image)
  ti.root.dense(ti.ijkl, (n_grid, n_grid, 3, steps)).place(refracted_image)
  ti.root.dense(ti.ijk, (n_grid, n_grid, 3)).place(target_img)
  ti.root.place(loss)
  ti.root.lazy_grad()


c = 340
# damping
alpha = 0.00000
inv_dx2 = inv_dx * inv_dx
dt = (math.sqrt(alpha * alpha + dx * dx / 3) - alpha) / c
learning_rate = 10.000000003


# TODO: there may by out-of-bound accesses here
@ti.func
def laplacian(t, i, j):
  return inv_dx2 * (-4 * p[t, i, j] + p[t, i, j - 1] + p[t, i, j + 1] +
                    p[t, i + 1, j] + p[t, i - 1, j])


@ti.func
def gradient(t, i, j):
  return 0.5 * inv_dx * ti.Vector(
      [p[t, i + 1, j] - p[t, i - 1, j], p[t, i, j + 1] - p[t, i, j - 1]])


@ti.kernel
def initialize():
  for i, j in initial:
    p[0, i, j] = initial[i, j]

@ti.kernel
def apply_wavegen(t: ti.i32):

  # def points_in_circle_np(radius, x0=0, y0=0, ):
  #     x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
  #     y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
  #     x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
  #     # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
  #     for x, y in zip(x_[x], y_[y]):
  #         yield x, y

  #for x,y in points_in_circle_np(50, 128, 128):

  #for i in range(n_grid):
  #  x = ti.cast(ti.floor(xy_circle[i][0]), ti.i32)
  #  y = ti.cast(ti.floor(xy_circle[i][1]), ti.i32)
  p[t - 1, 128, 128] = wave_gen[t - 1, 0]
  # p[t - 1, n_grid // 2 + 7, n_grid // 2 - 8] = wave_gen[t - 1, 1]
  # p[t - 1, n_grid // 2 - 8, n_grid // 2 - 10] = wave_gen[t - 1, 2]
  # p[t - 1, n_grid // 2 - 9, n_grid // 2 + 9] = wave_gen[t - 1, 3]


@ti.kernel
def apply_mask(t: ti.i32):
  #for i, j in mask:
  for i in range(n_grid):
    for j in range(n_grid):
      #ti.atom
      radius = (i - 128)**2 + (j - 128)**2
      if radius > 20**2 and radius < 30**2:
      #if (i > 108 and i < 148) and not (i > 118 and i < 138) or (j > 108 and j < 148) and not (j > 118 and j < 138) or ():
      #f (i < 118 or i > 138) and (j < 118 or j > 138) and (i > 108 or i < 148) and (j > 108 or j < 148):
        ti.atomic_sub(p[t-1, i, j], p[t-1, i, j] * (mask[i,j] / (mask[i,j] + 0.5)))
      
      # if mask[i,j] == 1:
      #   p[t-1, i, j] = 0.0
  
@ti.kernel
def fdtd(t: ti.i32):
  for i, j in height_gradient:
      laplacian_p = laplacian(t - 2, i, j)
      laplacian_q = laplacian(t - 1, i, j)
      p[t, i, j] = 2 * p[t - 1, i, j] + (
          c * c * dt * dt + c * alpha * dt
      ) * laplacian_q - p[t - 2, i, j] - c * alpha * dt * laplacian_p


@ti.kernel
def render_reflect():
  for i, j in height_gradient:
      grad = height_gradient[i, j]
      normal = ti.Vector.normalized(ti.Vector([grad[0], 1.0, grad[1]]))
      rendered[i, j] = normal[1]


@ti.kernel
def render_refract(t: ti.i32):
  for i, j in height_gradient:
    grad = height_gradient[i, j]

    scale = 2.0
    sample_x = i - grad[0] * scale
    sample_y = j - grad[1] * scale
    sample_x = ti.min(n_grid - 1, ti.max(0, sample_x))
    sample_y = ti.min(n_grid - 1, ti.max(0, sample_y))
    sample_xi = ti.cast(ti.floor(sample_x), ti.i32)
    sample_yi = ti.cast(ti.floor(sample_y), ti.i32)

    frac_x = sample_x - sample_xi
    frac_y = sample_y - sample_yi

    for k in ti.static(range(1)):
      refracted_image[i, j, k, t] = (1.0 - frac_x) * (
          (1 - frac_y) * bottom_image[sample_xi, sample_yi, k] +
          frac_y * bottom_image[sample_xi, sample_yi + 1, k]) + frac_x * (
              (1 - frac_y) * bottom_image[sample_xi + 1, sample_yi, k] +
              frac_y * bottom_image[sample_xi + 1, sample_yi + 1, k])


@ti.kernel
def compute_height_gradient(t: ti.i32):
  for i, j in height_gradient:  # Parallelized over GPU threads
      height_gradient[i, j] = gradient(t, i, j) * 1.0

@ti.kernel
def compute_loss():
  for i in range(n_grid):
    for j in range(n_grid):
      for k in range(1):
        # for t in range(150, steps):
        ti.atomic_add(
            loss,
            ti.sqr(target_img[i, j, k] - refracted_image[i, j, k, steps-1]) * (1 / n_grid**2))


@ti.kernel
def apply_grad():
  # gradient descent
  for i, j in mask.grad:
    mask[i, j] -= learning_rate * mask.grad[i, j]* 0.101
  # for i in range(steps):
  #   wave_gen[i,0] -= learning_rate * wave_gen.grad[i,0]
    # wave_gen[i,1] -= learning_rate * wave_gen.grad[i,1]
    # wave_gen[i,2] -= learning_rate * wave_gen.grad[i,2]
    # wave_gen[i,3] -= learning_rate * wave_gen.grad[i,3]


def forward(output=None):
  interval = vis_interval
  if output:
    os.makedirs(output, exist_ok=True)
    interval = output_vis_interval
  initialize()
  for t in range(2, steps):
    apply_wavegen(t)
    apply_mask(t)
    

    # for i in range(n_grid):
    #   for j in range(n_grid):
    #     p[t-1, i, j] = p[t-1, i, j] * (1.0 - mask[i,j])

    fdtd(t)
    # compute_height_gradient(t)
    # render_refract(t)
    if (t + 1) % interval == 0 and output is not None:
      compute_height_gradient(t)
      render_refract(t)
      img = refracted_image.to_numpy()
      img = cv2.resize(img[:, :, :, t], fx=4, fy=4, dsize=None)
      cv2.imshow('img', img)
      cv2.waitKey(1)
      if output:
        img = np.clip(img, 0, 255)
        cv2.imwrite(output + "/{:04d}.png".format(t), img * 255)

        img2 = height_gradient.to_numpy()
        img2 = cv2.resize(img2[:,:,0], fx=4, fy=4, dsize=None)
        img2 = np.clip(img2, 0, 255)
        cv2.imwrite(output + "/go{:04d}.png".format(t), img2 * 255)
        

        #cv2.imshow('img', img3)
        #cv2.waitKey(1)

    #if t == steps-1:
        img3 = mask.to_numpy()
        img3 = cv2.resize(img3, fx=4, fy=4, dsize=None)
        img3 = np.clip(img3, 0, 255)
        cv2.imwrite("water_renderer/mask.png", img3*20000)
        # cv2.imshow('img3', img3)
        # cv2.waitKey(1)

  compute_height_gradient(steps - 1)
  render_refract(t)
  compute_loss()

  #


def main():
  # initialization
  bot_img = cv2.imread('source4.jpg') / 255.0
  targ_img = cv2.imread('target4.jpg') / 255.0
  #target_img = np.array()
  #bot_img = bot_img / 255.0
  for i in range(256):
    for j in range(256):
      for k in range(3):
        bottom_image[i, j, k] = bot_img[i, j, k]
        target_img[i, j, k] = targ_img[i, j, k]

  #initial[n_grid // 2, n_grid // 2] = 99
  #forward('water_renderer/initial')
  #initial[20, n_grid // 2] = 100

  # for i in range(118, 138):
  #   for j in range(118, 138):
  #     mask[j, i] = 1

  for i in range(256):
    for j in range(256):
      if np.random.rand() < 0.1:
        radius = (i - 128)**2 + (j - 128)**2
        if radius > 20**2 and radius < 30**2:        
          mask[j, i] = 1.0


  for t in range(256):
    amplitude = np.sin(t / 2.0)
    wave_gen[t, 0] = amplitude

  #rr = (np.random.rand(256, 256) - 0.5) * 10.0

  #initial = rr

  #from adversarial import vgg_grad, predict

  for opt in range(100):
    with ti.Tape(loss):
      forward()

      # feed_to_vgg = np.zeros((224, 224, 3), dtype=np.float32)
      # # Note: do a transpose here
      # for i in range(224):
      #   for j in range(224):
      #     for k in range(3):
      #       feed_to_vgg[i, j, k] = refracted_image[i + 16, j + 16, 2 - k]

      # predict(feed_to_vgg)
      # grad = vgg_grad(feed_to_vgg)
      # for i in range(224):
      #   for j in range(224):
      #     for k in range(3):
      #       refracted_image.grad[i + 16, j + 16, k] = grad[i, j, 2 - k] * 0.001

      #feed_to_diff = np.zeros((224, 224, 3), dtype=np.float32)
      # Note: do a transpose here
      
      # for i in range(256):
      #   for j in range(256):
      #     for k in range(3):
      #       refracted_image.grad[i, j, k] = ti.sqr(target_img[i, j, k] - refracted_image[i, j, k]) * (1 / n_grid**2) * 1.0011 # - target_img[i, j, k]


            #feed_to_vgg[i, j, k] = refracted_image[i, j, k]

      # predict(feed_to_vgg)
      # grad = vgg_grad(feed_to_vgg)
      # for i in range(224):
      #   for j in range(224):
      #     for k in range(3):
      #       refracted_image.grad[i + 16, j + 16, k] = grad[i, j, 2 - k] * 0.001


      #refracted_image.grad = np.sqrt(target_img - bot_img) * (1 / n_grid**2) * 0.001
    

    print('Iter', opt)
    print('Iter', opt, ' Loss =', loss[None])

    apply_grad()

  forward('water_renderer/optimized')


if __name__ == '__main__':
  main()
