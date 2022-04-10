import numpy as np


A = [-1,-1,1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,-1,
-1,-1,1,-1,1,-1,-1,
-1,-1,1,-1,1,-1,-1,
-1,1,1,1,1,1,-1,
-1,1,-1,-1,-1,1,-1,
-1,1,-1,-1,-1,1,-1,
1,1,1,-1,1,1,1]

A1 = [-1,-1,-1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,-1,
-1,-1,-1,1,-1,-1,-1,
-1,-1,1,-1,1,-1,-1,
-1,-1,1,-1,1,-1,-1,
-1,1,-1,-1,-1,1,-1,
-1,1,1,1,1,1,-1,
-1,1,-1,-1,-1,1,-1,
-1,1,-1,-1,-1,1,-1]

B = [1,1,1,1,1,1,-1,
-1,1,-1,-1,-1,-1,1,
-1,1,-1,-1,-1,-1,1,
-1,1,-1,-1,-1,-1,1,
-1,1,1,1,1,1,-1,
-1,1,-1,-1,-1,-1,1,
-1,1,-1,-1,-1,-1,1,
-1,1,-1,-1,-1,-1,1,
1,1,1,1,1,1,-1]

C= [-1,-1,1,1,1,1,1,
-1,1,-1,-1,-1,-1,1,
1,-1,-1,-1,-1,-1,-1,
1,-1,-1,-1,-1,-1,-1,
1,-1,-1,-1,-1,-1,-1,
1,-1,-1,-1,-1,-1,-1,
1,-1,-1,-1,-1,-1,-1,
-1,1,-1,-1,-1,-1,1,
-1,-1,1,1,1,1,-1,
]

E = [1,1,1,1,1,1,1,
-1,1,-1,-1,-1,-1,1,
-1,1,-1,-1,-1,-1,-1,
-1,1,-1,1,-1,-1,-1,
-1,1,1,1,-1,-1,-1,
-1,1,-1,1,-1,-1,-1,
-1,1,-1,-1,-1,-1,-1,
-1,1,-1,-1,-1,-1,1,
1,1,1,1,1,1,1,
]


features = np.array(
    [
        A,
        B,
        C, 
        A1,
        E
    ])

print(features)
Atarget = [1,-1,-1,-1]
Btarget = [-1,1,-1,-1]
Ctarget = [-1,-1,1,-1]
Etarget = [-1,-1,-1,1]

target = np.array([Atarget,Btarget,Ctarget,Atarget,Etarget])

print("our model : ")
print(features, target)
print("\n \n")

weight = [np.zeros(63),np.zeros(63),np.zeros(63),np.zeros(63)]
bias = 0
alpha = 1
theta = 0.3
learning_rate = 0.1
epoch = 10

for i in range(epoch):
    print("*********************************  epoch :", i+1)

    for j in range(features.shape[0]):

        yin = [0,0,0,0]
        yout = [0,0,0,0]

        print(" =========== s",[j+1])

        actual = target[j]
        
        
      

        for k in range(4):
            sigma = 0
            for i in range(63):
                sigma += features[j][i] * weight[k][i]
            yin[k] = bias + sigma

            if(yin[k] > theta):
                yout[k] = 1
            elif(-theta <= yin[k] <= theta):
                yout[k] = 0
            else:
                yout[k] = -1
        
            if(yout[k]!= actual[k]):
                for i in range(63):
                    weight[k][i] += alpha * actual[k] * features[j][i]
                bias += alpha * actual[k]
        

        print( "Yin : " ,yin , "yout : ", yout , " actual : ",actual)
       
        for i in range(63):
            print( "w"+str(i)+"(new) : " ,weight[k][i] )

        print( "bias : " ,bias)
        

