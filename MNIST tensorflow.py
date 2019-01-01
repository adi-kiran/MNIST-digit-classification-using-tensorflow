import tensorflow as tf

#import datasets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

lr=0.5
epochs = 5
bth_size = 100 #training in batches of 100 images

#training data(input x=28x28pixels=784)
x = tf.placeholder(tf.float32, [None, 784])
#label has values of 10 numbers as 1 and 0s
y = tf.placeholder(tf.float32, [None, 10])

#weight and bias of input to hidden layer
Wih = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
bih = tf.Variable(tf.random_normal([300]), name='b1')
#weight and bias of hidden layer to the output
Who = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
bho = tf.Variable(tf.random_normal([10]), name='b2')

#output to the hidden layer using relu activation function
equn = tf.add(tf.matmul(x, Wih), bih)
h_op = tf.nn.relu(equn)

#calculate hidden layer output - using softmax activation which gives probability
y_ = tf.nn.softmax(tf.add(tf.matmul(h_op, Who), bho))

# cost function
y_ = tf.clip_by_value(y_, 1e-10, 0.9999999)     #to prevent log(0) 
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_)+ (1 - y) * tf.log(1 - y_), axis=1))

#gradient descent optimization
optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy)

#initialising the variables
init= tf.global_variables_initializer()

#calculate accuracy of network
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#scalar summary for accuracy
tf.summary.scalar('accuracy', acc)

merge=tf.summary.merge_all()

#session begins
with tf.Session() as sess:

        #call the initialising function
        sess.run(init)
        writer = tf.summary.FileWriter("output",sess.graph)
        tot_batch=int(len(mnist.train.labels)/bth_size)  #calculate no of batches to be run as we use mini batch concept
        for epoch in range(epochs):
            avgc= 0.0
            for i in range(tot_batch):
                xbatch, ybatch = mnist.train.next_batch(batch_size=bth_size)
                a,cost=sess.run([optimiser,cross_entropy],{x:xbatch, y:ybatch})
                avgc=avgc+(cost/tot_batch)
            print("Epoch:",(epoch + 1), "cost =",avgc)
            summary = sess.run(merge,{x: mnist.test.images, y: mnist.test.labels})
            writer.add_summary(summary, epoch)

        print("\n Training is complete!")
        print(sess.run(acc,{x:mnist.test.images, y:mnist.test.labels}))
        writer.close()