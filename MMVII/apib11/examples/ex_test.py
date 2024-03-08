import MMVII
import numpy as np

# Prepare data: see MMVII/MMVII-TestDir/Input/Saisies-MMV1/readme.md

dirData = '../../MMVII-TestDir/Input/Saisies-MMV1/'
imPath = '../../MMVII-TestDir/Input/EPIP/Tiny/ImL.tif'
dirOri = dirData + 'MMVII-PhgrProj/Ori/toto/'
calibFilePath = dirOri + 'Calib-PerspCentral-Foc-28000_Cam-PENTAX_K5.xml'
oriFilePath = dirOri + 'Ori-PerspCentral-IMGP4168.JPG.xml'

calib=MMVII.PerspCamIntrCalib.fromFile(calibFilePath)
print(calib.infoParam())

pp0=(1,2,3)
p0=(0,0)
p1=np.array((5,6))

box=MMVII.Box2di(p0,p1)
box2=MMVII.Box2di((-1,-1),(2,2))
box3=MMVII.Box3dr((-3,-3,-2.5),(3,3,2.5))

r=MMVII.Rect2(box2)


m=MMVII.Matrixr(10,10,MMVII.ModeInitImage.eMIA_Rand)
mf=MMVII.Matrixf(10,10,MMVII.ModeInitImage.eMIA_Rand)
a=np.array(m,copy=False)

im=MMVII.Im2Di.fromFile(imPath)
im_np=np.array(im,copy=False)

scpc=MMVII.SensorCamPC.fromFile(oriFilePath)
p=np.array([10,20,100])
diff=scpc.ground2ImageAndDepth(scpc.imageAndDepth2Ground(p))-p
print("diff = ",diff)


k=MMVII.Isometry3D((0,0,0),((1,0,0),(0,1,0),(0,0,1)))
r=MMVII.Rotation3D(((1,0,0),(0,1,0),(0,0,1)))

array = k.rot.array()

print(array)

l = MMVII.Isometry3D((0,0,0),array)

print (k == l)
print (k.rot == r)
print('test == pt:')
print (k.tr == (0,0,0))
print (k.tr == [0,0,0])
print (k.tr == np.array([0,0,0]))
print (np.array_equal(k.tr,(0,0,0)))

n = np.array( [ [1, 2, 3], [4, 5, 6] ] )
m = MMVII.Matrixr(n)

print( np.array_equal( (n-m), (m-n) ))

m = MMVII.Matrixr( [ [1, 2], [3, 4] ] )
n = np.array(m)

print(n@m)
(m@n).show()
(m@m).show()

p = ([8,9])
v = MMVII.Vectorr(p)
print(v)
print(m@p)
print(n@p)

