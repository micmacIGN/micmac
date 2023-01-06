from MMVII import *
import numpy as np


calib=PerspCamIntrCalib.fromFile("../MMVII-TestDir/Input/Ply/MMVII-PhgrProj/Ori/TestProjMesh/Calib-PerspCentral-Foc-14000_Cam-DCGX9.xml")
pp0=Pt3di(1,2,3)
p0=Pt2di(0,0)
p1=Pt2di(5,6)

box=Box2di(p0,p1)
box2=Box2di((-1,-1),(2,2))
box3=Box3dr((-3,-3,-2.5),(3,3,2.5))

r=Rect2(box2)


m=Matrixd(10,10,ModeInitImage.eMIA_Rand)
mf=Matrixf(10,10,ModeInitImage.eMIA_Rand)

a=np.array(m,copy=False)

for i in r:
    print(i,end=' ')
print("")

print(a)
m.show()


