import taichi as ti
#import seaborn as sns
#import numpy as np

ti.init(arch=ti.gpu)

n = 320
#pixels = ti.Vector.field(3, dtype=float, shape=(n, n))
pixels = ti.field(dtype=float, shape=(n, n))
max_iterations = 200
#palette = sns.light_palette("seagreen", as_cmap=True)
#colors = ti.Vector([palette(i)[0:3] for i in range(max_iterations)])
#print(colors)

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.kernel
def mandelbrot(t: float):
    for i, j in pixels:
        # slow down time
        t2 = 0.005 * t
        # sin has a range of (-1, 1)
        zoom = (-0.75 + ti.sin(t2) * 0.33)**8
        # normalize x and y to the window
        x = (i / n - 0.5)
        y = (j / n - 0.5)
        xy = ti.Vector([x, y])
        c = ti.Vector([-0.745, 0.186]) + xy*zoom
        v = ti.Vector([0.0, 0.0])
        iterations = 0
        max_iterations = 200
        while (v.norm() < 4 and iterations < max_iterations):
            v = complex_sqr(v) + c
            iterations += 1
        #print(colors[iterations])
        pixels[i, j] = 1 - iterations * 0.005
        #print(colors[iterations])
        #pixels[i, j] = ti.Vector([color[0], color[1], color[2]])

gui = ti.GUI("Mandelbrot", res=(n, n))
result_dir = "./results"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=60, automatic_build=False)

i = 0
while gui.running:
    mandelbrot(i)
    gui.set_image(pixels.to_numpy())
    gui.show()
    i += 1
#for i in range(1000):
#    mandelbrot(i)
#    video_manager.write_frame(pixels.to_numpy())
#    #gui.set_image(pixels.to_numpy())
#    #gui.show()
#while gui.running:
#    mandelbrot(1000000000)
#    gui.set_image(pixels.to_numpy())
#    gui.show()
#video_manager.make_video(gif=True, mp4=False)
