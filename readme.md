# Optic flow Lukas and Kanade's Optical flow

This is not my work, I simply translated the code into Python.

It takes two frames moments from each other.

Applies Gaussian derivative mask in x axis and y axis to the frame 1 and 2. This attains Ix and Iy.  

For It you just apply a normal Gaussian kernel and subtract the two frames.
I_x(q_i) = I_x
so on and so on

Then use this equation to calculate V_x:__
![alt text](https://github.com/sanny1/Opticflow/blob/master/equation.gif)
