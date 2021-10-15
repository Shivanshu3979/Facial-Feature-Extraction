import cv2
import os
import random
import numpy as np
class neuron:
  def __init__(self,val):
    self.val=val;
mini=999999;
total=0
class NeuralModel:
  def __init__(self,width,length,sizeHiddenLayers,outLabels=[],hiddenLayers=1,weight=0.2,bias=0):

    self.width=width;
    self.length=length;
    self.inpLayer=[0]*width*length;
    self.outputLabels=outLabels;
    self.hiddenLayers=hiddenLayers;
    self.sizeHiddenLayers=sizeHiddenLayers;
    self.weight={};
    self.biases={};
    self.activeLayer={};
    self.outlayer={};
    temp=[];


    self.finalweight={};
    self.finalbiases={};
    self.finalactiveLayer={};
    self.finaloutlayer={};
    
    
    
    
    for i in range(hiddenLayers):
      self.biases["layer"+str(i)]=[bias]*sizeHiddenLayers[i]
      self.activeLayer["layer"+str(i+1)]=[bias]*sizeHiddenLayers[i];
    self.biases["outLabels"]=[bias]*len(outLabels);

      
    self.N_C=width*length*sizeHiddenLayers[0];
    
    #weight matrix creaction
    for i in range(sizeHiddenLayers[0]):
      temp.append([weight]*length*width)
    self.weight["layer0"]=temp;
    temp=[];
    for i in range(hiddenLayers-1):
      self.N_C+=sizeHiddenLayers[i]*sizeHiddenLayers[i+1];
      for j in range(sizeHiddenLayers[i+1]):
        temp.append([weight]*sizeHiddenLayers[i+1]);
      self.weight["layer"+str(i+1)]=temp;
      
    temp=[];
    for i in range(len(outLabels)):
      temp.append([weight]*sizeHiddenLayers[-1]);
    self.weight["layer"+str(hiddenLayers)]=temp;
     

    

    
    self.N_C += sizeHiddenLayers[-1]*len(outLabels);
    B_C=sum(sizeHiddenLayers)+len(outLabels)
    print("Total Weigths Count:\t",self.N_C);
    print("Total Biases Count:\t",B_C);
    print("total variable/Epoch:\t",B_C+self.N_C);

  def save(self):
    t1=self.weight;
    t2=self.biases;
    t3=self.activeLayer;
    t4=self.outlayer;
    if len(self.finalweight)>1:
      for i in t1:
        t1[i]=np.array(t1[i])
        self.finalweight[i]=(self.finalweight[i]+t1[i])/2

      for i in t2:
        t2[i]=np.array(t2[i])
        self.finalbiases[i]=(self.finalbiases[i]+t2[i])/2

      for i in t3:
        t3[i]=np.array(t3[i])
        self.finalactiveLayer[i]=(self.finalactiveLayer[i]+t3[i])/2

      self.weight=t1;
    else:
      for i in t1:
        t1[i]=np.array(t1[i])
      self.finalweight=t1;
      for i in t2:
        t2[i]=np.array(t2[i])
      self.finalbiases=t2;
      for i in t3:
        t3[i]=np.array(t3[i])
      self.finalactiveLayer=t3;
      for i in t4:
        t4[i]=np.array(t4[i])
      self.finaloutlayer=t4;

      
  def cost(self,t_param,req_label):
    sum=0;
    for i in t_param:
      if i!=req_label:
        sum+=abs(t_param[i]-0);
        #print(abs(t_param[i]-0))
      else:
        sum+=abs(t_param[i]-1);
        #print(abs(t_param[i]-1))
    return sum;

  def t_weight(self,index,lr):
    global total;
    for i in range(len(self.weight["layer0"])):
      for j in range(len(self.weight["layer0"][i])):
        temp=round(random.uniform(0,1),1)
        total+=1
        while (self.weight["layer0"][i][j]==temp):
          temp=round(random.uniform(0,1),1)
          
        self.weight["layer0"][i][j]=temp
      
    for i in range(len(self.weight["layer1"])):
      for j in range(len(self.weight["layer1"][i])):
        total+=1
        temp=round(random.uniform(0,1),1)
        while (self.weight["layer1"][i][j]==temp):
          temp=round(random.uniform(0,1),1)
        self.weight["layer1"][i][j]=temp

    for i in range(len(self.weight["layer2"])):
      for j in range(len(self.weight["layer2"][i])):
        total+=1
        temp=round(random.uniform(0,1),1)
        while (self.weight["layer2"][0][j]==temp):
          temp=round(random.uniform(0,1),1)
        self.weight["layer2"][0][j]=temp 
    
    return index+1;


  
  def train(self,lr,image,label):
    global mini
    global total
    index=0;
    image=cv2.resize(image,(self.width,self.length));
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("image",image)
    for i in image:
      for j in i:
        self.inpLayer[index]=(j/255)
        index+=1;
    temp=0;
    rate=0;
    
    index=0;
    previousPer=0;
    print("work done(%):",end=" ");
    while True:
      for i in range(len(self.weight["layer0"])):
        for j in range(len(self.weight["layer0"][i])):
          temp+=self.weight["layer0"][i][j]*self.inpLayer[j]
        self.activeLayer["layer1"][i]=self.biases["layer0"][i]+temp/len(self.inpLayer);
        temp=0;
      #print("Layer1 Complete")
      
      for i in range(len(self.weight["layer1"])):
        for j in range(len(self.weight["layer1"][i])):
          temp+=self.weight["layer1"][i][j]*self.activeLayer["layer1"][j]
        self.activeLayer["layer2"][i]=self.biases["layer1"][i]+temp/len(self.activeLayer["layer1"]);
        temp=0;
      #print("Layer2 Complete")
      
      for i in range(len(self.weight["layer2"])):
        for j in range(len(self.weight["layer2"][i])):
          temp+=self.weight["layer2"][i][j]*self.activeLayer["layer2"][j]
        self.outlayer[self.outputLabels[i]]=self.biases["outLabels"][i]+temp/len(self.activeLayer["layer2"]);
        temp=0;
      #print("Output Layer Complete")
      #print(label+":",self.cost(self.outlayer,label))


      if self.cost(self.outlayer,label)<mini:
        self.save();
        mini=self.cost(self.outlayer,label);
        print("  minimum cost=",mini,end="");
      else:
        rate+=lr;
      index=self.t_weight(index,lr)+1
      total+=1;
      #print(((index*100)/self.N_C),"\b")
      if index > self.N_C/100:
        if int((index*100)/self.N_C)>previousPer:
          print("  $"*int(int((index*100)/self.N_C)/3),((index*100)/self.N_C),end="%  ");
          previousPer=int((index*100)/self.N_C)

      #os.system("cls") 
      if index>self.N_C:
        print("\ncost:",mini,"outlayer:",model.finaloutlayer,"\n\n")
        break
      print("\rcurrent:",self.cost(self.outlayer,label),end="");

       
    

if __name__=='__main__':
  
  model=NeuralModel(width=64,
                    length=64,
                    outLabels=['male','female'],
                    hiddenLayers=2,
                    sizeHiddenLayers=(36,36))
  directory="train/male/"
  epoch=1;
  for i in os.listdir("train/male"):
    image=cv2.imread(directory+i)
    print(i)
    for j in range(epoch):
      print("epoch",j+1)
      model.train(lr=0.1,image=image,label="male")

    break;
  print(total)
  
                    
                    
