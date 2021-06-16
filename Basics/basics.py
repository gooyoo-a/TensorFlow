

# gooyoo




# !pip install tensorflow;
import tensorflow as tf;




# Initialization

x = tf.constant(4, shape = (1, 1), dtype = tf.float64);
print(x);


x = tf.constant([[1, 2, 3], [4, 5, 6]], shape = (2, 3));
print(x);


x = tf.eye(4);
print(x);


x = tf.ones((4, 5));
print(x);


x = tf.zeros((3, 2, 4));
print(x);


x = tf.random.uniform((3, 3), minval = 0, maxval = 1);
print(x);


x = tf.random.normal((4, 4), mean = 0, stddev = 1);
print(x);
print(tf.cast(x, dtype = tf.float64));


x = tf.range(7);
print(x);
x = tf.range(start = 0, limit = 20, delta = 2);
print(x);




# Math

x = tf.constant([1, 2, 3]);
y = tf.constant([9, 8, 7]);

z1 = tf.add(x, y);
print(z1);
z2 = x + y;
print(z2);
print(z1 == z2);


z1 = tf.subtract(x, y);
print(z1);
z2 = x - y;
print(z2);
print(z1 == z2);


z1 = tf.divide(x, y);
print(z1);
z2 = x / y;
print(z2);
print(z1 == z2);


z1 = tf.multiply(x, y);
print(z1);
z2 = x * y;
print(z2);


z = tf.tensordot(x, y, axes=1);
print(z);


z = x ** 5;
print(z);


x = tf.random.normal((2, 3));
y = tf.random.normal((3, 2));

z1 = tf.matmul(x, y);
print(z1);
z2 = x @ y;
print(z2);
print(z1 == z2);




# Indexing

x = tf.constant([0, 1, 1, 2, 3, 1, 7, 3]);

print(x[:]);
print(x[1:]);
print(x[1:3]);
print(x[::2]);
print(x[::-1]);


indices = tf.constant([0, 4, 6]);
x_indices = tf.gather(x, indices);
print(x_indices);


x = tf.constant([[1, 2], [3, 4], [5, 6]]);

print(x[0, :]);
print(x[0:2, :]);




# Reshaping

x = tf.range(9);
print(x);
x = tf.reshape(x, (3, 3));
print(x);
x = tf.transpose(x, perm = [1, 0]);
print(x);





