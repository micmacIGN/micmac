import mmv2
startState = mmv2.cMemManager_CurState()
if (1):
  im=mmv2.cIm2Du1.FromFile("ex/image.tif") #req template cIm2Du1
  #im.DIm().ToFile("ex/out.tif") #req template cDataIm2Du1
  print("Nb objects created: ", mmv2.cMemManager_CurState().NbObjCreated())

mmv2.cMemManager_CheckRestoration(startState)
#del b
#mmv2.cMemManager_CheckRestoration(startState)

aSz = mmv2.Pt2di(100,20)
aFileIm = mmv2.cDataFileIm2D_Create("ex/toto.tif",mmv2.eTyNums_eTN_U_INT1,aSz,1);
aIDup = mmv2.cIm2Du1(aSz)
aIDup.Write(aFileIm,mmv2.Pt2di(0,0))

im=mmv2.cIm2Du1.FromFile("ex/image.tif") #here image type must be exact
im2 = im.GaussDeZoom(3)
im2.DIm().ToFile("ex/out.tif")

#ImageScale python remake
mScale = 2
mDilate = 1.0
aFileIn = mmv2.cDataFileIm2D_Create("ex/image.tif",True)
aImIn = mmv2.cIm2Dr4(aFileIn.Sz())
aImIn.Read(aFileIn,mmv2.Pt2di(0,0))

aImOut = aImIn.GaussDeZoom(mScale,3,mDilate)
aFileOut = mmv2.cDataFileIm2D_Create("ex/out.tif",aFileIn.Type(),aImOut.DIm().Sz(),1)
aImOut.Write(aFileOut,mmv2.Pt2di(0,0))
