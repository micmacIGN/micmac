import mmv2
startState = mmv2.cMemManager_CurState()
if (1):
  b=mmv2.cIm2Du8.FromFile("ex/image.tif") #req template cIm2Du8
  #b.DIm().ToFile("ex/out.tif") #req template cDataIm2Du8
  print("Nb objects created: ", mmv2.cMemManager_CurState().NbObjCreated())

mmv2.cMemManager_CheckRestoration(startState)
#del b
#mmv2.cMemManager_CheckRestoration(startState)

