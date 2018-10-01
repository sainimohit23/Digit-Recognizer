import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

camera = cv2.VideoCapture(0)
blueLower = np.array([100, 36, 20], dtype = "uint8")
blueUpper = np.array([231, 86, 90], dtype = "uint8")




def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized






#Placeholders
def create_placeholders(n_H, n_W, n_C, n_Y):
    
    X = tf.placeholder(tf.float32, shape= [None, n_H, n_W, n_C])
    Y = tf.placeholder(tf.float32, shape= [None, n_Y])
    
    return X, Y




#Variables or Weights or filters
def initialize_parameters():
    
    W1 = tf.get_variable('W1', shape=[3, 3, 1, 8], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable('W2', shape=[6, 6, 8, 16], initializer= tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[8], initializer= tf.zeros_initializer())
    b2 = tf.get_variable('b2', shape=[16], initializer= tf.zeros_initializer())
    
    parameters = {'W1': W1,
                  'W2': W2,
                  'b1': b1,
                  'b2': b2
                  }
    
    return parameters




#Forward Propagation
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

#Layer 1
    Z1 = tf.nn.conv2d(X, W1, strides= [1, 1, 1, 1],padding='SAME')
    
    batch_mean1, batch_var1 = tf.nn.moments(X,[0, 1, 2])
    scale1 = tf.Variable(tf.ones([8]))
    beta1 = tf.Variable(tf.zeros([8]))
    BN1 = tf.nn.batch_normalization(Z1, batch_mean1, batch_var1,beta1, scale1, 0.0000001)
    
    A1 = tf.nn.relu(BN1 + b1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 3, 3, 1], strides= [1, 3, 3, 1], padding='SAME')
    
#Layer 2    
    Z2 = tf.nn.conv2d(P1, W2, strides= [1, 1, 1, 1],padding='SAME')
    
    batch_mean2, batch_var2 = tf.nn.moments(X,[0, 1, 2])
    scale2 = tf.Variable(tf.ones([16]))
    beta2 = tf.Variable(tf.zeros([16]))
    BN2 = tf.nn.batch_normalization(Z2, batch_mean2, batch_var2,beta2, scale2, 0.0000001)
    
    A2 = tf.nn.relu(BN2 + b2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 6, 6, 1], strides= [1, 6, 6, 1], padding='SAME')
    
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 20)
    
    Z4 = tf.contrib.layers.fully_connected(Z3, 20)
    Z5 = tf.nn.softmax(tf.contrib.layers.fully_connected(Z4, 10, activation_fn=None))
    
    
    return Z5


X, Y = create_placeholders(28, 28, 1, 10)
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters)




with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
    
    while True:
        ret, frame = camera.read()
        
        frame = resize(frame, 400)
        blue = cv2.inRange(frame, blueLower, blueUpper)
        blue = cv2.GaussianBlur(blue, (3,3), 0)
    
        _, cnts, _ = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pred = None
        if len(cnts)> 0:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            term = 0
            if h>w:
                term = h
            else:
                term = w
            
            roi = blue[y:y+term,x:x+term]
            roi = np.pad(roi, 3, 'minimum')
            roi = cv2.resize(roi,(28,28))        
            roi = np.reshape(roi,(1,28,28,1))
            pred = sess.run(Z3, feed_dict={X : roi})
        
        cv2.putText(frame,str(np.argmax(pred)), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow('res', frame)
        cv2.imshow('blu', blue)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()
    
    
