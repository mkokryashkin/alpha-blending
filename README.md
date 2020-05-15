# MIPT Assembly course Alpha-blending
Alpha blending is the process of combining a translucent foreground color with a background color, thereby producing a new color blended between the two. 

So for each pixel we do the same thing 3 times, because we need to apply formula to each color. We can optimize this with SIMD instructions. You can access them directly from C/C++ code with Intel Intrinsics.

Here is the alpha-blending result picture:
![blended](/pictures/composed.bmp)

And, as we can see, SIMD-optimized alpha-blending is a lot more fast than version written completely in C++:
![stats](/diagram.png)
