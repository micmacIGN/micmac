import mmv2

def checkMem():
  import sys
  print("Nb objects created: ", mmv2.cMemManager_CurState().NbObjCreated())
  startState = mmv2.cMemManager_CurState()
  def tmp():
    im=mmv2.cIm2Du1.FromFile("ex/image.tif")
  tmp()
  print("Nb objects created: ", mmv2.cMemManager_CurState().NbObjCreated())
  mmv2.cMemManager_CheckRestoration(startState)

  b=mmv2.cIm2Du1.FromFile("ex/image.tif")
  print("Nb objects created: ", mmv2.cMemManager_CurState().NbObjCreated())
  #del b
  try:
    mmv2.cMemManager_CheckRestoration(startState)
    print("cMemManager_CheckRestoration OK")
  except:
    print("cMemManager_CheckRestoration failed")
    print("Error: ",sys.exc_info()[0])

def createImage():
  aSz = mmv2.Pt2di(100,20)
  aFileIm = mmv2.cDataFileIm2D_Create("ex/toto.tif",mmv2.eTyNums_eTN_U_INT1,aSz,1);
  aIDup = mmv2.cIm2Du1(aSz)
  aIDup.Write(aFileIm,(0,0)) #points params can be given as sequences

def quickDezoom():
  im=mmv2.cIm2Du1.FromFile("ex/image.tif") #here image type must be exact
  im2 = im.GaussDeZoom(3)
  im2.DIm().ToFile("ex/out.tif")

def imageScale():
  mScale = 2
  mDilate = 1.0
  aFileIn = mmv2.cDataFileIm2D_Create("ex/image.tif",True)
  aImIn = mmv2.cIm2Dr4(aFileIn.Sz())
  aImIn.Read(aFileIn,(0,0))

  aImOut = aImIn.GaussDeZoom(mScale,3,mDilate)
  aFileOut = mmv2.cDataFileIm2D_Create("ex/out.tif",aFileIn.Type(),aImOut.DIm().Sz(),1)
  aImOut.Write(aFileOut,(0,0))

def imgNumpyRawData():
  from PIL import Image
  import numpy as np
  im=mmv2.cIm2Du1.FromFile("ex/image.tif")
  array = im.DIm().toArray()
  img = Image.fromarray(array)
  img.show()

def PIL2mmv2Img():
  from PIL import Image
  import numpy as np
  path="ex/png.png"
  pil_image = Image.open(path).convert('L')
  pil_image_array = np.array(pil_image)
  
  im = mmv2.cIm2Du1( (5,5) )
  d = im.DIm()
  d.setRawData(pil_image_array)
  d.ToFile("ex/out.tif")
  
  array = d.toArray()
  img = Image.fromarray(array)
  img.show()

def to8Bit():
  import numpy as np
  im = mmv2.cIm2Dr4.FromFile("ex/exfloat32.tif")
  mat = im.DIm().toArray()
  mat2 = (255*(mat-np.min(mat))/(np.max(mat)-np.min(mat))).astype('uint8')
  imOut = mmv2.cIm2Du1( im.DIm().Sz() )
  imOut.DIm().setRawData(mat2)
  imOut.DIm().ToFile("ex/to8bits.tif")
