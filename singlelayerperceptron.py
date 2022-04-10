import numpy as np


features = np.array(
    [
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ])

print(features)

target = np.array([1,-1, -1, -1])

print("our model : ")
print(features, target)
print("\n \n")

weight = [0, 0]
bias = 0
alpha = 1
theta = 0.3
thetanegative = -0.3
learning_rate = 0.1
epoch = 2

 
for i in range(epoch):
    print("*********************************  epoch :", i+1)

    for j in range(features.shape[0]):

        print(" =========== s",[j+1])

        actual = target[j]
        
        x1 = features[j][0]
        x2 = features[j][1]

        print( "X1 : " ,x1,"X2 : " ,x2 , " target : "+ str(actual))

        yin = bias + (x1 * weight[0]) + (x2 * weight[1])

        

        yout = -2 
        if(yin > theta):
            yout = 1
        elif(thetanegative <= yin <= theta):
            yout = 0
        else:
            yout = -1
  
        print( "Yin : " ,yin , "yout : ", yout)
       
        if(yout != actual):
            print("new weights ----------- : ")
            weight[0] += alpha * actual * x1
            weight[1] += alpha * actual * x2
            bias += alpha * actual
  
       

        print( "w1(new) : " ,weight[0])
        print( "w2(new) : " ,weight[1])
        print( "bias : " ,bias)
        

