import MMVII
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
    TP photogrammetry
    Data from MurSaintMartin
    Prepare data: see MMVII/MMVII-TestDir/Input/Saisies-MMV1/readme.md
"""

dirData = '../../MMVII-TestDir/Input/Saisies-MMV1/'
dirOri = dirData + 'MMVII-PhgrProj/Ori/toto/'
dirPts = dirData + 'MMVII-PhgrProj/PointsMeasure/Saisies_MMVII/'
imNames = ('IMGP4168.JPG', 'IMGP4169.JPG')

# 3D points
pts3d = MMVII.SetMesGCP.fromFile(dirPts + 'MesGCP-FromV1-Ground-Pts3D.xml')

# 2D points
pts2d = {}
for imName in imNames:
    pts2d[imName] = MMVII.SetMesPtOf1Im.fromFile(dirPts + 'MesIm-' + imName + '.xml')

########################################################
#     1 - Read Calib, apply it
########################################################
imName1 = imNames[0]
# Show 2d points
plt.title('Read 2d points')
allX, allY = zip(*[ mes.pt for mes in pts2d[imName1].measures()])
img = mpimg.imread(dirData + imName1)
plt.imshow(img)
plt.scatter(allX, allY, 30)
plt.show()

# apply calibration
calibPath = dirOri + 'Calib-PerspCentral-Foc-28000_Cam-PENTAX_K5.xml'
calib = MMVII.PerspCamIntrCalib.fromFile(calibPath)

dist = calib.dir_Dist()
pp2i = calib.mapPProj2Im()
i2pp = pp2i.mapInverse()
inv_dist = MMVII.DataInvertOfMapping2D(calib.dir_DistInvertible())

for mes in pts2d[imName1].measures():
    pt = mes.pt
    print('Pt im: ', pt)
    print(' -> central with disto: ', i2pp.value(pt))
    print(' -> central no disto: ', inv_dist.value(i2pp.value(pt)))
    print(' -> lig/col no disto: ', pp2i.value(inv_dist.value(i2pp.value(pt))))


def myPP2I(calib: MMVII.PerspCamIntrCalib, pt_central: tuple) -> tuple:
    # TODO, see doc 3.2
    pass

def myDist(calib: MMVII.PerspCamIntrCalib, pt_central: tuple) -> tuple:
    # TODO, see doc ...
    pass

print('Test myPP2I:')
for mes in pts2d[imName1].measures():
    pt = mes.pt
    print('Pt im: ', pt)
    pt_central = i2pp.value(pt)
    print(' -> central with disto: ', pt_central)
    print(' -> my lig/col with disto: ', myPP2I(dist, pt_central))


print('Test myDist:')
for mes in pts2d[imName1].measures():
    pt = mes.pt
    print('Pt im: ', pt)
    pt_central_nodist = inv_dist.value(i2pp.value(pt))
    print(' -> central no disto: ', pt_central_nodist)
    print(' -> my central with disto: ', myDist(dist, pt_central_nodist))

########################################################
#     2 - Read Ori, projection
########################################################

cam1= MMVII.SensorCamPC.fromFile(dirOri + 'Ori-PerspCentral-'+imName1+'.xml')

# project ground points on image
allX, allY,_ = zip(*map(cam1.ground2ImageAndDepth,[mes.pt for mes in pts3d.measures()]))

plt.title('Projected 3d points')
img = mpimg.imread(dirData + imName1)
plt.imshow(img)
plt.scatter(allX, allY, 30)
plt.show()

def myGround2ImageAndDepth(camPose: MMVII.Isometry3D,
                           calib: MMVII.PerspCamIntrCalib,
                           pt3d: tuple) -> tuple:
    # TODO, see doc 3.2
    pass 

print('Test myGround2ImageAndDepth')
for pt in [mes.pt for mes in pts3d.measures()]:
    print('Proj MM: ', cam1.ground2ImageAndDepth(pt))
    print('My proj: ', myGround2ImageAndDepth(cam1.pose, cam1.internalCalib, pt))

########################################################
#     3 - Pseudo-intersection
########################################################

# Second image
imName2 = imNames[1]
cam2= MMVII.SensorCamPC.fromFile(dirOri + 'Ori-PerspCentral-'+imName2+'.xml')

# Compute 2d points without disortion
pts2d_nodist = {}
for imName in pts2d.keys():
    pts2d_nodist[imName] = {}
    for mes in pts2d[imName].measures():
        name = mes.namePt
        pt = mes.pt
        pts2d_nodist[imName][name] = list(inv_dist.value(i2pp.value(pt)))


# TODO: compute 3d coordinates of common points
def myPseudoIntersect(cam1: MMVII.SensorCamPC, pt2d1: tuple,
                      cam2: MMVII.SensorCamPC, pt2d2: tuple) -> tuple:
    # TODO, see doc 3.2
    pass 

print('Test myPseudoIntersect')
# apply to common points between im1 and im2
for mes in pts2d[imName1].measures():
    ptName = mes.namePt
    pt1 = mes.pt
    if not ptName in [mes.namePt for mes in pts3d.measures()] :
        continue
    if not pts2d[imName2].nameHasMeasure(ptName):
        continue
    pt2 = pts2d[imName2].measuresOfName(ptName)
    # for now, there is no pts3d.measuresOfName()
    ground_coords = {mes.namePt:mes.pt for mes in pts3d.measures()}[ptName]
    print(ptName, ': GND =', ground_coords)
    print('Pseudo-intersection: ', myPseudoIntersect(cam1, pt1, cam2, pt2))

########################################################
#     4 - Orientation from known points
########################################################

# recompute orientation from 4  2d/3d points
# TODO

