import mm3d
import os

"""
Equivalent to TestCam
"""
c=mm3d.CamOrientFromFile("examples_py/Orientation-IMG_5564.tif.xml")

p=mm3d.Pt2dr(1000,1000)
prof=1

print('''//   R3 : "reel" coordonnee initiale
//   L3 : "Locale", apres rotation
//   C2 :  camera, avant distortion
//   F2 : finale apres Distortion
//   M2 : coordonnees scannees 
//
//       Orientation      Projection      Distortion      Interne mm
//   R3 -------------> L3------------>C2------------->F2------------>M2
//''')
print("Focale ",c.Focale())
print("M2 ",p," ---> F2 ",c.NormM2C(p)," --->C2 ",c.F2toC2(c.NormM2C(p))," ---> R3 ",c.ImEtProf2Terrain(c.NormM2C(p),prof))


"""
Rotations
"""

l=[1,0,0,0,1,0,0,0,1]
print(l)
r=mm3d.list2rot(l)
print(r)
ll=mm3d.rot2list(r)
print(ll)

mm3d.createIdealCamXML (100.0, mm3d.Pt2dr(1000,1000), mm3d.Pt2di(2000,2000), "Ori", "Im001", "cam", r, 200.0, 1200)

li = mm3d.getFileSet(".",".*.py")

r=mm3d.quaternion2rot(0.706421209674, 0.000595506000, -0.002847643999, -0.707785709674)
l=mm3d.rot2list(r)

"""
Read homol file
"""

pack = mm3d.ElPackHomologue.FromFile("examples_py/IMG_5564.dat")
print(pack.size())
list_homol=pack.getList()
for h in list_homol[0:10]:
   print(h.P1(),h.P2())

aPackOut=mm3d.ElPackHomologue()
aCple=mm3d.ElCplePtsHomologues(mm3d.Pt2dr(10,10),mm3d.Pt2dr(20,20));
aPackOut.Cple_Add(aCple);
aPackOut.StdPutInFile("examples_py/homol.dat");

"""
Exceptions
"""
"""
try:
  c=mm3d.cTD_Camera("Ori-FishEyeBasic/Orientation-Calibration_geo_14_001_01_015000.thm.dng_G.tif.xml")
except:
  print("argh")
"""


try:
  c=mm3d.CamOrientFromFile("not_existing.xml")
except RuntimeError as e:
  print(e)


"""
Create couples file for Tapioca (for 2 parallel cameras)
"""
aRel = mm3d.cSauvegardeNamedRel()

#line part
for c in range(1,3):
  for f in range(1,10):
    imA="cam{}_img{:05}".format(c,f)
    imB="cam{}_img{:05}".format(c,f+1)
    aRel.Cple().append(mm3d.cCpleString(imA,imB))

#parallel part
for f in range(1,11):
  imA="cam1_img{:05}".format(f)
  imB="cam2_img{:05}".format(f)
  aRel.Cple().append(mm3d.cCpleString(imA,imB))

mm3d.MakeFileXML_cSauvegardeNamedRel(aRel,"cpl.xml")


"""
Lecture d'une liste de triplets  
"""
aWDir = 'examples_py/'

# Read data manager
aCMA = mm3d.cCommonMartiniAppli()
aNM = aCMA.NM(aWDir)

# Read triplets
nameTri = aNM.NameTopoTriplet(False)
print("nameTri: ",nameTri)
aXml_TopoTriplet = mm3d.StdGetFromSI_Xml_TopoTriplet(nameTri)
aTris = aXml_TopoTriplet.Triplets()
nbTri = aTris.size()
print("Found ",nbTri," triplets")
