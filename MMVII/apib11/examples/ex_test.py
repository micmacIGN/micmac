from MMVII import *
import numpy as np


calib=PerspCamIntrCalib.fromFile("../../MMVII-TestDir/Input/Ply/MMVII-PhgrProj/Ori/TestProjMesh/Calib-PerspCentral-Foc-14000_Cam-DCGX9.xml")
pp0=Pt3di(1,2,3)
p0=Pt2di(0,0)
p1=Pt2di(5,6)

box=Box2di(p0,p1)
box2=Box2di((-1,-1),(2,2))
box3=Box3dr((-3,-3,-2.5),(3,3,2.5))

r=Rect2(box2)


m=Matrixr(10,10,ModeInitImage.eMIA_Rand)
mf=Matrixf(10,10,ModeInitImage.eMIA_Rand)

a=np.array(m,copy=False)

im=Im2Di.fromFile("../../MMVII-TestDir/Input/EPIP/Tiny/ImL.tif")
im_np=np.array(im,copy=False)

scpc=SensorCamPC.fromFile("../../MMVII-TestDir/Input/Ply/MMVII-PhgrProj/Ori/TestProjMesh/Ori-PerspCentral-P1056160.JPG.xml")
p=(10,20,100)
diff=scpc.ground2ImageAndDepth(scpc.imageAndDepth2Ground(p))-p
print("diff = ",diff)


k=Isometry3D((0,0,0),((1,0,0),(0,1,0),(0,0,1)))
r=Rotation3D(((1,0,0),(0,1,0),(0,0,1)))

array = k.rot.array()

print(array)

l = Isometry3D((0,0,0),array)

print (k == l)
print (k.rot == r)
print (k.tr == Pt3dr(0,0,0))
