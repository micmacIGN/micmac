from MMVII import *

"""
    TP photogra
    Data from MurSaintMartin
"""


pts3d = {
  'I-of-I': (0.754045221758523621, 0.496427193197002514, 0.431416432371162006),
  'Acute': (0.898750585424819226, 1.10409507249266081, 0.444680581722013746),
  'Finger-1': (1.31771333753878528, 0.989188464947304724, 0.370627429238321537),
  'Stone-6': (0.150697933447584281, 1.10818337179321613, 0.773464897817899177),
  'Stone-7': (0.256396378205514475, 2.35318759021626445, 2.23887785258234207),
  'Grille': (1.40207519252831658, 2.1945458898399548, 0.40382985984277503)
}

pts2d = {
  'I-of-I': (430.52658272147653, 227.090326619720145),
  #'u-of-You': (495.717626755149013, 454.825546116027397),
  #'Acute': (431.735281413390453, 510.222991095907105),
  #'Finger-1': (316.373353330086218, 471.678176822389275),
  #'Stone-6': (667.402028214032839, 479.731591690534685),
  'Stone-7': (1284.05120739189965, 981.021140912616147),
  'Grille': (362.602015084684353, 1078.96450589692108)
}

imName = 'IMGP4168.JPG'
calib = PerspCamIntrCalib.fromFile('Calib-PerspCentral-Foc-28000_Cam-PENTAX_K5.xml')
cam = SensorCamPC(imName, Isometry3D.identity(), calib)

# recompute orientation with 4 points
# TODO


# compare with oritentation in reference file
cam = SensorCamPC.fromFile('Ori-PerspCentral-IMGP4168.JPG.xml')


predicts = {}
for name, pt in pts3d.items():
    predicts[name] = cam.ground2ImageAndDepth(pt)

# Show predictions
allX = [p.x for p in predicts.values()]
allY = [p.y for p in predicts.values()]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(imName)
plt.imshow(img)
plt.scatter(allX, allY, 3)



