from MMVII import *
import numpy as np

# Prepare data: see MMVII/MMVII-TestDir/Input/Saisies-MMV1/readme.md

dirData = '../../MMVII-TestDir/Input/Saisies-MMV1/'
calib=PerspCamIntrCalib.fromFile(dirData + 'MMVII-PhgrProj/Ori/toto/Calib-PerspCentral-Foc-28000_Cam-PENTAX_K5.xml')
print(calib.infoParam())

pp0=(1,2,3)
p0=(0,0)
p1=np.array((5,6))

box=Box2di(p0,p1)
box2=Box2di((-1,-1),(2,2))
box3=Box3dr((-3,-3,-2.5),(3,3,2.5))

r=Rect2(box2)


m=Matrixr(10,10,ModeInitImage.eMIA_Rand)
mf=Matrixf(10,10,ModeInitImage.eMIA_Rand)
a=np.array(m,copy=False)

im=Im2Di.fromFile("../../MMVII-TestDir/Input/EPIP/Tiny/ImL.tif")
im_np=np.array(im,copy=False)

scpc=SensorCamPC.fromFile(dirData + 'MMVII-PhgrProj/Ori/toto/Ori-PerspCentral-IMGP4168.JPG.xml')
p=np.array([10,20,100])
diff=scpc.ground2ImageAndDepth(scpc.imageAndDepth2Ground(p))-p
print("diff = ",diff)


k=Isometry3D((0,0,0),((1,0,0),(0,1,0),(0,0,1)))
r=Rotation3D(((1,0,0),(0,1,0),(0,0,1)))

array = k.rot.array()

print(array)

l = Isometry3D((0,0,0),array)

print (k == l)
print (k.rot == r)
print('test == pt:')
print (k.tr == (0,0,0))
print (k.tr == [0,0,0])
print (k.tr == np.array([0,0,0]))
print (np.array_equal(k.tr,(0,0,0)))

n = np.array( [ [1, 2, 3], [4, 5, 6] ] )
m = Matrixr(n)

print( np.array_equal( (n-m), (m-n) ))

m = Matrixr( [ [1, 2], [3, 4] ] )
n = np.array(m)

print(n@m)
(m@n).show()
(m@m).show()

p = ([8,9])
v = Vectorr(p)
print(v)
print(m@p)
print(n@p)

