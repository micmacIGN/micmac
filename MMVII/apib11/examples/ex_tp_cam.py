from MMVII import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
    TP photogra
    Data from MurSaintMartin
"""

dirData = '../../MMVII-TestDir/Input/Saisies-MMV1/'

# from Ground-Pts3D.xml
pts3d = {
  'I-of-I': (0.754045221758523621, 0.496427193197002514, 0.431416432371162006),
  'Acute': (0.898750585424819226, 1.10409507249266081, 0.444680581722013746),
  'Finger-1': (1.31771333753878528, 0.989188464947304724, 0.370627429238321537),
  'Stone-6': (0.150697933447584281, 1.10818337179321613, 0.773464897817899177),
  'Stone-7': (0.256396378205514475, 2.35318759021626445, 2.23887785258234207),
  'Grille': (1.40207519252831658, 2.1945458898399548, 0.40382985984277503)
}

# from GroundMeasure.xml
pts2d = {
  'IMGP4168.JPG' : {
      'I-of-I': (430.52658272147653, 227.090326619720145),
      'u-of-You': (495.717626755149013, 454.825546116027397),
      'Acute': (431.735281413390453, 510.222991095907105),
      'Finger-1': (316.373353330086218, 471.678176822389275),
      'Stone-6': (667.402028214032839, 479.731591690534685),
      'Stone-7': (1284.05120739189965, 981.021140912616147),
      'Grille': (362.602015084684353, 1078.96450589692108)
  },
  'IMGP4169.JPG' : {
      'I-of-I': (475.161154787777321, 516.404520623212079),
      'u-of-You': (538.999416573275425, 732.995393503311675),
      'Acute': (475.850364860953221, 802.548038024082189),
      'Finger-1': (362.721106696891582, 789.307120972098687),
      'Stone-6': (706.139274179003905, 741.525951210447488)
  }
}

########################################################
#     1 - Read Calib, apply it
########################################################


imName1 = 'IMGP4168.JPG'

# Show 2d points
plt.title('Read 2d points')
allX, allY = zip(*pts2d[imName1].values())
img = mpimg.imread(dirData + imName1)
plt.imshow(img)
plt.scatter(allX, allY, 30)
plt.show()

# apply calibration
calib = PerspCamIntrCalib.fromFile(dirData + 'Ori-Ground-MMVII/Calib-PerspCentral-Foc-28000_Cam-PENTAX_K5.xml')

dist = calib.dir_Dist()
sten = calib.calibStenPerfect()
inv_sten = sten.mapInverse()
inv_dist = DataInvertOfMapping2D(calib.dir_DistInvertible())

for name, pt in pts2d[imName1].items():
    print('Pt im: ', pt)
    print(' -> central with disto: ', inv_sten.value(pt))
    print(' -> central no disto: ', inv_dist.value(inv_sten.value(pt)))
    print(' -> lig/col no disto: ', sten.value(inv_dist.value(inv_sten.value(pt))))


def mySten(calib: PerspCamIntrCalib, pt_central: tuple) -> tuple:
    # TODO, see doc 3.2
    pass

def myDist(calib: PerspCamIntrCalib, pt_central: tuple) -> tuple:
    # TODO, see doc ...
    pass

print('Test mySten:')
for name, pt in pts2d[imName1].items():
    print('Pt im: ', pt)
    pt_central = inv_sten.value(pt)
    print(' -> central with disto: ', pt_central)
    print(' -> my lig/col with disto: ', mySten(dist, pt_central))


print('Test myDist:')
for name, pt in pts2d[imName1].items():
    print('Pt im: ', pt)
    pt_central_nodist = inv_dist.value(inv_sten.value(pt))
    print(' -> central no disto: ', pt_central_nodist)
    print(' -> my central with disto: ', myDist(dist, pt_central_nodist))



########################################################
#     2 - Read Ori, projection
########################################################

cam1= SensorCamPC.fromFile(dirData + 'Ori-Ground-MMVII/Ori-PerspCentral-'+imName1+'.xml')

# project ground points on image
allX, allY,_ = zip(*map(cam1.ground2ImageAndDepth,pts3d.values()))

plt.title('Projected 3d points')
img = mpimg.imread(dirData + imName1)
plt.imshow(img)
plt.scatter(allX, allY, 30)
plt.show()

def myGround2ImageAndDepth(camPose: Isometry3D, calib: PerspCamIntrCalib, pt3d: tuple) -> tuple:
    # TODO, see doc 3.2
    pass 

for name, pt in pts3d.items():
    print('Proj MM: ', cam1.ground2ImageAndDepth(pt))
    print('My proj: ', myGround2ImageAndDepth(cam1.pose, cam1.internalCalib, pt))

########################################################
#     3 - Pseudo-intersection
########################################################

# Second image
imName2 = 'IMGP4169.JPG'
cam2= SensorCamPC.fromFile(dirData + 'Ori-Ground-MMVII/Ori-PerspCentral-'+imName2+'.xml')

# Compute 2d points without disortion
pts2d_nodist = {}
for imName in pts2d.keys():
    pts2d_nodist[imName] = {}
    for name, pt in pts2d[imName].items():
        pts2d_nodist[imName][name] = list(inv_dist.value(inv_sten.value(pt)))


# TODO: compute 3d coordinates of common points
def myPseudoIntersect(cam1: SensorCamPC, pt2d1: tuple, cam2: SensorCamPC, pt2d2: tuple) -> tuple:
    # TODO, see doc 3.2
    pass 

print('Pseudo-intersection:')
# apply to common points between im1 and im2
for ptName, pt1 in pts2d[imName1].items():
    if not ptName in pts3d :
        continue
    if not ptName in pts2d[imName2] :
        continue
    pt2 = pts2d[imName2][ptName]
    print(ptName, ': GND =', pts3d[ptName])
    print('Pseudo-intersection: ', myPseudoIntersect(cam1, pt1, cam2, pt2))


########################################################
#     4 - Orientation from known points
########################################################

# recompute orientation from 4  2d/3d points
# TODO



