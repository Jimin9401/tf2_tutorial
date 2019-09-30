import tensorflow as tf


hello=tf.constant("hello, tensorflow!")

print(hello)

a=tf.constant(10)
b=tf.constant(5)

c=tf.add(a,b)



sess=tf.Session()

print(sess.run(hello))
print(sess.run([a,b,c]))




X=tf.placeholder(tf.float32,[None,3])

x_data=[[1,2,3],[4,5,6]]


W=tf.Variable(tf.random_normal([3,2]))
b=tf.Variable(tf.random_normal([2,1]))


expr=tf.matmul(X,W)+b

sess.run(tf.global_variables_initializer())

print(x_data)
print(sess.run(W))
print(sess.run(b))

print(sess.run(expr,feed_dict={X:x_data}))

sess.close()





















