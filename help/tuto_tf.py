#-------------------------------------------------------------------------------
# Name:        tuto_tf
# Purpose:
#
# Author:      thomasbl
#
# Created:     25/10/2016
# Copyright:   (c) thomasbl 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import tensorflow as tf

#how to see tensorflow operation
def seeTF():
    # one 3x3 image with 2 channels
    input = tf.Variable(tf.random_normal([1,3,3,2]))
    # one 3x3 filter with 2 channels
    filter = tf.Variable(tf.random_normal([3,3,2,1]))

    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        print("input")
        print(input.eval())
        print("filter")
        print(filter.eval())
        print("result")
        result = sess.run(op)
        result2 = sess.run(op2)
        print(result)
        print(result2)

    #if we have following matrix image:
    #     i1 i2 i3
    # I = i4 i5 i6
    #     i7 i8 i9
    #with the following filters:
    #      w1 w2 w3         f1 f2 f3
    # W =  w4 w5 w6 and F = f4 f5 f6
    #      w7 w8 w9         f7 f8 f9
    #with padding VALID the center of the filter is set to the center
    # . . .                 . . . . .
    # . x . with 5x5 image: . x x x .
    # . . .                 . x x x .
    #                       . x x x .
    #                       . . . . .
    #so the result will be equal to r where
    # r = np.sum( I*W + I*F ) here it's r = np.sum( I[0,:,:,0]*W[:,:,0,0] + I[0,:,:,1]*W[:,:,1,0])
    #with padding SAME we keep the same dim as output so we have:
    # 0 0 0 0 0
    # 0 x x x 0
    # 0 x x x 0
    # 0 x x x 0
    # 0 0 0 0 0
    #so the result will be equal to
    #     r1 r2 r3
    # R = r4 r5 r6
    #     r7 r8 r9
    #where r1 = np.sum( I[0,0:2,0:2,0]*W[1:,1:,0,0] + I[0,0:2,0:2,1]*W[1:,1:,1,0])

def tuto_ts_conv2d():
    #input image WxH in RGB -> 3 input channels
    # nb_filters = nb_output_channels
    #padding VALID - perform valid convolution
    #padding SAME - keep the same output dimension than the input
    input = tf.Variable(tf.random_normal([1,3,3,5]))
    filter = tf.Variable(tf.random_normal([1,1,5,1]))
    # strides -> [1, stride, stride, 1]
    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    #3x3 image and 1x1 filter each have 5 channels
    # input -> [nb_image, image_width, image_height, nb_channels]
    # filter -> [filter_width, filter_height, nb_input_channels, nb_filters]
    #output = nb_images x width x height x nb_filters(here nb_filters = nb_output_channels)
    # -> here 1x3x3x1

    input = tf.Variable(tf.random_normal([1,3,3,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,1]))
    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    #image 3x3, 3x3 filter, 5 channels
    #output = 1x1x1x1, value is the sum of the 9,5-element dot product, 45-element dot product

    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,1]))
    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
    #output = 1x3x3x1

    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,1]))
    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    #output = 1x5x5x1

    #with multiple filters, here 7
    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,7]))
    op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    #output = 1x5x5x7

    #strides 2,2
    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,7]))
    op = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
    #output = 1x3x3x7
    # x . x . X
    # . . . . .
    # x . x . x
    # . . . . .
    # x . x . x

    #now with 10 images
    input = tf.Variable(tf.random_normal([10,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,7]))
    op = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
    #output = 10x3x3x7