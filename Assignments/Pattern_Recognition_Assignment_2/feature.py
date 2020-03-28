import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def extract_feature_dataset2b(folder):
    images_data = load_images_from_folder(folder)

    for i in range(0,len(images_data)):
        completeName = os.path.join(folder+"/feature",str(i))
        f = open(completeName,"w")
        for j in range(0,len(images_data[i]),32):
            for k in range(0,len(images_data[i][j]),32):
                red = np.zeros(8,dtype='int')
                green = np.zeros(8,dtype='int')
                blue = np.zeros(8,dtype='int')
                sumred=0
                sumgreen=0
                sumblue=0
                count_pixel=0
                for m in range(j,j+32):
                    for n in range(k,k+32):
                        if m<len(images_data[i]) and n<len(images_data[i][j]):
                            count_pixel+=1
                            x=images_data[i][m][n][0]
                            sumred+=x
                        else:
                            x = int(sumred/count_pixel)
                        loop=0
                        while x:
                            red[7-loop]+=(x%2)
                            x/=2
                            loop+=1

                        if m<len(images_data[i]) and n<len(images_data[i][j]):
                            x=images_data[i][m][n][1]
                            sumgreen+=x
                        else:
                            x = int(sumgreen/count_pixel)
                        loop=0
                        while x:
                            green[7-loop]+=(x%2)
                            x/=2
                            loop+=1

                        if m<len(images_data[i]) and n<len(images_data[i][j]):
                            x=images_data[i][m][n][2]
                            sumblue+=x
                        else:
                            x = int(sumblue/count_pixel)
                        loop=0
                        while x:
                            blue[7-loop]+=(x%2) 
                            x/=2
                            loop+=1

                for p in range(0,len(red)):
                    f.write(str(red[p])+" ")
                for p in range(0,len(red)):
                    f.write(str(green[p])+ " ")
                for p in range(0,len(red)):
                    f.write(str(blue[p])+" ")
                f.write("\n")
        f.close()


def extract_feature_dataset2C(folder):
    images_data = load_images_from_folder(folder)
    for i in range(0,len(images_data)):
        completeName = os.path.join(folder+"/feature",str(i))
        f = open(completeName,"w")
        for j in range(0,len(images_data[i]),7):
            for k in range(0,len(images_data[i][j]),7):
                
                #Calculate the mean of a path
                sumgrey = 0.0
                num = 0
                for m in range(j,j+7):
                    for n in range(k,k+7):
                        if m<len(images_data[i]) and n<len(images_data[i][j]):
                            sumgrey+=images_data[i][m][n][0]
                            num+=1
                mean = sumgrey/num

                #Calculate the variance of a path
                sumVariance = 0.0
                num = 0
                for m in range(j,j+7):
                    for n in range(k,k+7):
                        if m<len(images_data[i]) and n<len(images_data[i][j]):
                            x = images_data[i][m][n][0]
                            sumVariance+=((x-mean)*(x-mean))
                        num+=1
                variance = sumVariance/num

                f.write(str(mean)+" "+str(variance))
                f.write("\n")
        f.close()


# extract_feature_dataset2b('2b/group06/test/aqueduct')
# extract_feature_dataset2b('2b/group06/test/industrial_area')
# extract_feature_dataset2b('2b/group06/test/patio')

# extract_feature_dataset2b('2b/group06/train/aqueduct')
# extract_feature_dataset2b('2b/group06/train/industrial_area')
# extract_feature_dataset2b('2b/group06/train/patio')

extract_feature_dataset2C('2c/group06/Train')
extract_feature_dataset2C('2c/group06/Test')
