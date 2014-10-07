/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include "XML_GEN/all_tpl.h"

#if (ELISE_QT_VERSION >= 4)
    #ifdef Int
        #undef Int
    #endif

    #include <QApplication>
    #include <QInputDialog>

    #include "general/visual_mainwindow.h"
#endif


class cAppliSake
{
  public:
    cAppliSake(int argc,char ** argv);
    int Exe();
    void ReadType(const std::string & aType);
    void InitDefValFromType();
    void ShowParamVal();

  private:
    std::string   mImPat, mDir, mMaskIm, mDirMEC, mPyr, mDirOrtho, mModeOri, mOriFileExtension;
    double        mZInc, mZMoy, mStepF, mRegul ;
    int           mSzW, mZoomI, mZoomF;
    bool          mCalcMEC, mEZA, mExe;
    std::string   mInstruct;
    std::string   mImgs;
    eTypeSake     mCorrelGeomType;
    std::string   mStrCorrelGeomType; //temporary string for mCorrelGeomType, used for LArgMain
    bool          mModeHelp;
    cInterfChantierNameManipulateur * mICNM;
    const cInterfChantierNameManipulateur::tSet * mSetIm;
    int           mNbIm, mNbStepsMEC;
    bool          mUseMastIm;
    std::string   mMastIm;
};

cAppliSake::cAppliSake(int argc,char ** argv) :
  mImPat            (""),
  mDir              (""),
  mMaskIm           (""),
  mDirMEC           ("MEC-Sake/"),
  mPyr              ("Pyram/"),
  mDirOrtho         (""),
  mModeOri          ("GRID"),
  mOriFileExtension ("GRI"),
  mZInc             (1000.0),
  mZMoy             (1000.0),
  mStepF             (0.5),
  mRegul           (0.01),
  mSzW              (1),
  mZoomI            (32),
  mZoomF            (1),
  mCalcMEC          (true),
  mEZA              (false),
  mExe              (true),
  mUseMastIm        (false),
  mMastIm           ("")
{
#if(ELISE_QT_VERSION >= 4)
  QApplication app(argc, argv);
  if (MMVisualMode)
  {
    std::cout<<"MMVisualMode"<<std::endl;
    LArgMain LAM;
    LAM << EAMC(mStrCorrelGeomType,"Correlation geometry",eSAM_None,ListOfVal(eNbTypeVals));

    std::vector <cMMSpecArg> aVA = LAM.ExportMMSpec();

    cMMSpecArg aArg = aVA[0];

    list<string> liste_valeur_enum = listPossibleValues(aArg);

    QStringList items;
    list<string>::iterator it=liste_valeur_enum.begin();
    for (; it != liste_valeur_enum.end(); ++it)
      items << QString((*it).c_str());

    setStyleSheet(app);

    bool ok = false;
    QInputDialog myDialog;
    QString item = myDialog.getItem(NULL, app.applicationName(), QString (aArg.Comment().c_str()), items, 0, false, &ok);

    if (ok && !item.isEmpty())
      mStrCorrelGeomType = item.toStdString();
    else
      return;

    ReadType(mStrCorrelGeomType);
  }
  else
  {
    ELISE_ASSERT(argc >= 2,"Not enough arguments");
    ReadType(argv[1]);
  }
#else
    ELISE_ASSERT(argc >= 2,"Not enough arguments");
    ReadType(argv[1]);
#endif

  InitDefValFromType();

  Box2dr            aBoxClip, aBoxTerrain;

  std::string       mModeGeomIm;
  std::string       mModeGeomMnt;

  ElInitArgMain
  (
    argc,argv,
    LArgMain()
              << EAMC(mStrCorrelGeomType,"Correlation geometry (one of the allowed enumerated values)")
              << EAMC(mImPat,"Images' path (Directory+Pattern)", eSAM_IsPatFile)
              << EAMC(mOriFileExtension,"Orientation file extension (Def=GRI)", eSAM_IsExistDirOri),
    LArgMain()
              << EAM(mModeOri,"ModeOri", true, "Orientation type (GRID or RTO; Def=GRID)", eSAM_NoInit)
              //<< EAM(mMastIm,"MasterIm",true,"Master image is mandatory if the correlation mode geometry is GeomIm", eSAM_IsExistFileRP)
              << EAM(mMaskIm,"Mask",true,"Mask file")
              << EAM(mSzW,"SzW",true,"Correlation window size (Def=2, equiv 5x5)")
              << EAM(mZMoy,"ZMoy",true,"Average value of Z (Def=1000.0)")
              << EAM(mZInc,"ZInc",true,"Initial uncertainty on Z (Def=1000.0)")
              << EAM(mRegul,"ZRegul",true,"Regularization factor (Def=0.01")
              << EAM(mStepF,"ZPas",true,"Quantification step (Def=0.5)")
              //<< EAM(mZoomInit,"ZoomI",true,"Initial Zoom (Def=32)")//
              << EAM(mZoomF,"ZoomF",true,"Final zoom (Def=1)")
              << EAM(aBoxClip,"BoxClip",true,"Define computation area (Def=[0,0,1,1], means full area)", eSAM_Normalize)
              << EAM(aBoxTerrain,"BoxTerrain",true,"Define computation area [Xmin,Ymin,Xmax,Ymax]")
              << EAM(mEZA,"EZA",true,"Export absolute values for Z (Def=false)", eSAM_IsBool)
              << EAM(mExe,"Exe",true,"Execute command (Def=true)", eSAM_IsBool)
              << EAM(mCalcMEC,"DoMEC",true,"Compute the matching (Def=true)", eSAM_IsBool)
              << EAM(mDirMEC,"DirMEC",true,"Results subdirectory (Def=MEC-Sake/")
              << EAM(mDirOrtho,"DirOrtho",true,"Orthos subdirectory if OrthoIm (Def=Ortho-${DirMEC})")
  );


#if (ELISE_windows)
  replace(mImPat.begin(), mImPat.end(), '\\', '/' );
#endif
  SplitDirAndFile(mDir,mImgs,mImPat);


  mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
  mSetIm = mICNM->Get(mImgs);
  mNbIm = mSetIm->size();
  ELISE_ASSERT((mNbIm>=2)|mUseMastIm,"Not enough images in pattern!!");

  MakeFileDirCompl(mDirMEC);

  if (mDirOrtho=="")
    mDirOrtho = "Ortho-" + mDirMEC;
  MakeFileDirCompl(mDirOrtho);

  std::string aFileMM = "MM-Sake.xml";

  //~ ShowParamVal();

  mInstruct =  MM3dBinFile_quotes("MICMAC")
            +  ToStrBlkCorr( Basic_XML_MM_File(aFileMM) )
            + std::string(" WorkDir=") + mDir
            + std::string(" +ImPat=")  + QUOTE(mImgs)
            + std::string(" +OriFileExt=")    + mOriFileExtension
            + std::string(" +SzW=")    + ToString(mSzW)
            + std::string(" +ZInc=") + ToString(mZInc)
            + std::string(" +ZMoy=") + ToString(mZMoy)
            + std::string(" +ZPasF=")    + ToString(mStepF)
            + std::string(" +ZRegul=") + ToString(mRegul)
           ;

  if (mZoomF<mZoomI)
  {
    if (log2(mZoomF)!=int(log2(mZoomF)))
    {
      std::ostringstream err_mess;
      err_mess<<"Incorrect value for ZoomF (must be a power of 2 and < ";
      err_mess<<mZoomI<<")";
      ELISE_ASSERT(false, err_mess.str().c_str());
    }
  }
  else
  {
    ELISE_ASSERT(false, "Value for ZoomF too high");
  }


  ELISE_ASSERT(EAMIsInit(&mModeOri), "ModeOri not given (mandatory param)");
  ELISE_ASSERT(EAMIsInit(&mZMoy), "ZMoy not given (mandatory param)");
  ELISE_ASSERT(EAMIsInit(&mZInc), "ZInc not given (mandatory param)");

  if (mModeOri=="GRID")  mModeGeomIm="eGeomImageGrille";
  else if (mModeOri=="RTO")  mModeGeomIm="eGeomImageRTO";
  else  ELISE_ASSERT(false,"Unknown orientation mode (must be GRID or RTO)!");
  mInstruct = mInstruct + std::string(" +ModeGeomIm=") + mModeGeomIm;

  mModeGeomMnt="eGeomMNTEuclid";
  //~ if (mStrCorrelGeomType=="GeomIm")
  //~ {
    //~ if (mMastIm!="")
    //~ {
      //~ mModeGeomMnt="eGeomMNTFaisceauIm1ZTerrain_Px1D";
      //~ if (! EAMIsInit(&mDirMEC))   mDirMEC = "MM-Sake-Img-" + StdPrefix(mMastIm) +ELISE_CAR_DIR;
//~
      //~ mInstruct =  mInstruct + std::string(" +UseMasterIm=true")
                             //~ + std::string(" +MasterIm=") + mMastIm;
    //~ }
    //~ else ELISE_ASSERT(false,"Master image not given (MasterIm is mandatory if GeomIm)");
  //~ }

  mInstruct = mInstruct + std::string(" +ModeGeomMNT=") + mModeGeomMnt;


  mInstruct = mInstruct + std::string(" +DirMEC=") + mDirMEC;

  if (mStrCorrelGeomType=="OrthoIm")
  {
    mInstruct = mInstruct + std::string(" +CalcOrtho=true");
    mInstruct = mInstruct + std::string(" +DirOrtho=") + mDirOrtho;
    std::cout<< "DirOrtho: *" << mDirOrtho << "*"<<std::endl;
  }

  if (EAMIsInit(&mMaskIm))
  {
    std::string aNameMask;

    ELISE_ASSERT(mDir==DirOfFile(mMaskIm),"Mask image not in working directory!!!"); //mDir: mask's directory
    SplitDirAndFile(mDir,aNameMask,mMaskIm); //mMaskIm: mask's full path (dir+name)
    if (IsPostfixed(aNameMask)) aNameMask = StdPrefixGen(aNameMask); //aNameMask: mask's filename without postfix


    mInstruct =  mInstruct + " +UseMask=true"
              + std::string(" +Mask=")  + aNameMask;
  }

   if (EAMIsInit(&aBoxClip))
  {
    mInstruct = mInstruct + " +UseClip=true "
              +  std::string(" +X0Clip=") + ToString(aBoxClip._p0.x)
              +  std::string(" +Y0Clip=") + ToString(aBoxClip._p0.y)
              +  std::string(" +X1Clip=") + ToString(aBoxClip._p1.x)
              +  std::string(" +Y1Clip=") + ToString(aBoxClip._p1.y) ;
  }

  if (EAMIsInit(&aBoxTerrain))
  {
    mInstruct = mInstruct + " +UseBoxTer=true "
              +  std::string(" +X0Ter=") + ToString(aBoxTerrain._p0.x)
              +  std::string(" +Y0Ter=") + ToString(aBoxTerrain._p0.y)
              +  std::string(" +X1Ter=") + ToString(aBoxTerrain._p1.x)
              +  std::string(" +Y1Ter=") + ToString(aBoxTerrain._p1.y) ;
  }

  mNbStepsMEC = 1 + round_ni(log2(mZoomI/mZoomF)) +1; // number of MEC steps (if no duplicate of zoom)

  mInstruct = mInstruct + std::string(" +CalcMEC=") + (mCalcMEC ? "true" : "false")
                        + std::string(" +EZA=") + (mEZA ? "true" : "false")
                        + std::string(" +ZoomF=") + ToString(mZoomF)
                        + std::string(" +NbSteps=") + ToString(mNbStepsMEC)
                        + std::string(" +Exe=") + (mExe ? "true" : "false")
                        ;

  ShowParamVal();

  //~ std::cout<<"*********************"<<std::endl;
  //~ std::cout<<"MicMac call: ** "<<mInstruct<<" **"<<std::endl;
  std::cout<<"** "<<mInstruct<<" **"<<std::endl;
  //~ std::cout<<"*********************"<<std::endl;



}

int cAppliSake::Exe()
{
  if (!mExe) return 0;
  int aRes = TopSystem(mInstruct.c_str());
  if (!MMVisualMode) ShowParamVal();
  return aRes;
}

void cAppliSake::ReadType(const std::string & aType)
{
    mStrCorrelGeomType = aType;
    StdReadEnum(mModeHelp,mCorrelGeomType,mStrCorrelGeomType,eNbTypeVals);
}


void cAppliSake::InitDefValFromType()
{
  switch (mCorrelGeomType)
  {
    case eGeomTer:
      //~ mSzW = 1;
      mSzW = 2;
      mRegul = 0.01;
      mZoomF = 1;
      break;

    //~ case eGeomIm :
      //~ mSzW = 1;
      //~ mRegul = 0.01;
      //~ mZoomF = 1;
      //~ break;

     case eOrthoIm :
      mSzW = 2;
      mRegul = 0.05;
      //~ mZoomF = 2;
      mZoomF = 1;
      break;

    case eNbTypeVals :
      break;

    };
}

void cAppliSake::ShowParamVal()
{
  std::cout << "********** SAKE - SAtellite Kit for Elevation ********"<<std::endl;
  std::cout << "********************Parameters************************"<<std::endl;
  std::cout << "*   Correl geometry: "<< mStrCorrelGeomType <<std::endl;
  std::cout << "*   Number of images: " << mNbIm << std::endl;
  std::cout << "*   Correl window size: " << 2*mSzW+1 << "x"  << 2*mSzW+1 << " (SzW=" << mSzW << ")" << std::endl;
  std::cout << "*   Correl step: " << mStepF << std::endl;
  std::cout << "*   Regularization term: " << mRegul << std::endl;
  std::cout << "*   Final DeZoom: " << mZoomF << std::endl;
  std::cout << "*   Number of correlation steps: " << mNbStepsMEC << std::endl;
  std::cout << "*   MEC subdirectory: " << mDirMEC << std::endl;
  if (mStrCorrelGeomType=="OrthoIm")  std::cout << "*   Orthos subdirectory: " << mDirOrtho << std::endl;
  std::cout << "******************************************************"<<std::endl;
}


int Sake_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);
    cAppliSake aAppli(argc,argv);
    int aRes = aAppli.Exe();
    return aRes;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �   la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement, �   l'utilisation,  �   la modification et/ou au
développement et �   la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �   des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �   leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�   l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
