# Optic flow Lukas and Kanade's Optical flow

This is not my work, I simply translated the code into Python.

It takes two frame and calculates the velocity vector in x and y direction.

Applies Gaussian derivative mask in x axis and y axis to the frame 1 and 2. This attains I<sub>x</sub> and I<sub>y</sub>.  

For It you just apply a normal Gaussian kernel and subtract the two frames.
I<sub>x</sub>(q<sub>i</sub>) = I<sub>x</sub>

so on and so on

Then use this equation to calculate V<sub>x</sub> and V<sub>y</sub>:

![alt text](https://github.com/sanny1/Opticflow/blob/master/equation.gif)

And here is the Image:

![alt text](https://github.com/sanny1/Opticflow/blob/master/Opticflow_2_frames.png)
