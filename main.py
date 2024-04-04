import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os,cv2

#function
def plot_images(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.90,hspace=0.35)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray)
        plt.title(titles[i],size=12)
        plt.xticks(())
        plt.yticks(())

directory="D:/practice/face_recognition/faces/"
Y=[];X=[];target_names=[]
person_id=0;h=w=300
n_samples=0
class_names=[]
for person_name in os.listdir(directory):
    dir_path=directory+person_name+"/"
    class_names.append(person_name)
    for img_name in os.listdir(dir_path):
        img_path=dir_path+img_name
        img=cv2.imread(img_path)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        resized_img=cv2.resize(gray,(h,w))
        v=resized_img.flatten()
        X.append(v)
        n_samples=n_samples+1
        Y.append(person_id)
        target_names.append(person_name)
    person_id=person_id+1

#transforming list into numpy array
Y=np.array(Y)
X=np.array(X)
target_names=np.array(target_names)
n_features=X.shape[1]
print(Y.shape,X.shape,target_names.shape)
print("Numbers of samples: ",n_samples)
print("Total dataset size is: ")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)

#tarin and tes split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.25,random_state=42)

n_component=150
print("Extracting the top %d eigenfaces from %d faces"% (n_component,X_train.shape[0]))

#PCA
pca=PCA(n_components=n_component,svd_solver='randomized',whiten=True).fit(X_train)
eigenfaces=pca.components_.reshape((n_component,h,w))
eigenface_titles=["eigenface %d"% i for i in range (eigenfaces.shape[0])]
plot_images(eigenfaces,eigenface_titles,h,w)
plt.show()
print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)
print(X_train_pca.shape,X_test_pca.shape)
lda=LinearDiscriminantAnalysis()
lda.fit(X_train_pca,Y_train)
X_train_lda=lda.transform(X_train_pca)
X_test_lda=lda.transform(X_test_pca)
print("done")

#Training with multi-layer perception
clf=MLPClassifier(random_state=1,hidden_layer_sizes=(10,10),max_iter=1000,verbose=True).fit(X_train_lda,Y_train)
print("Model weights:")
model_info=[coef.shape for coef in clf.coefs_]
print(model_info)

Y_prediction=[];Y_probability=[]
for test_face in X_test_lda:
    probability=clf.predict_proba([test_face])[0]
    class_id=np.where(probability==np.max(probability))[0][0]
    Y_prediction.append(class_id)
    Y_probability.append(np.max(probability))

Y_prediction=np.array(Y_prediction)

prediction_titles=[]
true_positive=0
for i in range(Y_prediction.shape[0]):
    true_name=class_names[Y_test[i]]
    prediction_name=class_names[Y_prediction[i]]
    result='pred: %s,pr: %s \ntrue: %s' % (prediction_name,str(Y_probability[i])[0:3], true_name)
    prediction_titles.append(result)
    if true_name==prediction_name:
        true_positive=true_positive+1
print("Accuracy:",true_positive*100/Y_prediction.shape[0])

plot_images(X_test,prediction_titles,h,w)
plt.show()