import tensorflow as tf
import math


def axis_angle_from_quaternion(quaternion, batch_size):
    """ converts a batch of quaternion outputs to axis-angle format """
    qx = tf.slice(quaternion, [0, 1], [batch_size, 1])
    qy = tf.slice(quaternion, [0, 2], [batch_size, 1])
    qz = tf.slice(quaternion, [0, 3], [batch_size, 1])
    qw = tf.slice(quaternion, [0, 0], [batch_size, 1])
    # normalize quaternion
    q_length = tf.sqrt(tf.reduce_sum([tf.pow(qw, 2), tf.pow(qx, 2), tf.pow(qy, 2), tf.pow(qz, 2)], 0))
    qw = tf.divide(qw, q_length)
    qx = tf.divide(qx, q_length)
    qy = tf.divide(qy, q_length)
    qz = tf.divide(qz, q_length)
    normalized_quaternion = tf.stack([qw, qx, qy, qz])
    # calculate angle and axis
    angle = tf.divide(tf.multiply(tf.multiply(2.0, tf.acos(qw)), 180.0), math.pi)
    axis_x = tf.divide(qx, tf.sqrt(tf.subtract(1.0, tf.multiply(qw, qw))))
    axis_y = tf.divide(qy, tf.sqrt(tf.subtract(1.0, tf.multiply(qw, qw))))
    axis_z = tf.divide(qz, tf.sqrt(tf.subtract(1.0, tf.multiply(qw, qw))))
    return angle, axis_x, axis_y, axis_z, normalized_quaternion


def accuracies(predictedAngle, predictedX, predictedY, predictedZ, realAngle, realX, realY, realZ, error):
    """ returns the ops required to calculate the accuracies of a predicted quaternion (angle, axis, both) with a specified error """
    # angle
    angle_diff = tf.abs(tf.subtract(predictedAngle, realAngle))
    correct_angle = tf.less_equal(angle_diff, error)
    accuracy_angle = tf.reduce_mean(tf.cast(correct_angle, tf.float32))
    # axis
    axis_angle_radians = tf.acos(
        tf.reduce_sum([tf.multiply(predictedX, realX), tf.multiply(predictedY, realY), tf.multiply(predictedZ, realZ)],
                      0))
    axis_angle_degrees = tf.divide(tf.multiply(axis_angle_radians, 180.0), math.pi)
    correct_axis = tf.less_equal(axis_angle_degrees, error)
    accuracy_axis = tf.reduce_mean(tf.cast(correct_axis, tf.float32))
    # both
    accuracy_both = tf.reduce_mean(tf.cast(tf.logical_and(correct_angle, correct_axis), tf.float32))
    accuracies = tf.stack([accuracy_angle, accuracy_axis, accuracy_both])
    return accuracies, angle_diff, axis_angle_degrees