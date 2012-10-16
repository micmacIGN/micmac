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

#include "general/all.h"
#include "private/all.h"

#include "XML_GEN/all.h"
#include "XML_GEN/all_tpl.h"
using namespace NS_ParamChantierPhotogram;


template <class Type> void VerifIn(const Type & aV,const Type * aTab,int aNb, const std::string & aMes)
{
     for (int aK=0 ; aK< aNb ; aK++)
     {
         if (aTab[aK] == aV)
         {
	       return;
         }
     }
     std::cout << "\n\nIn context " << aMes << "\n";
     std::cout << "With value  " << aV << "\n";
     
     std::cout << "Allowed  values are   : "  ; 
     for (int aK=0 ; aK< aNb ; aK++)
         std::cout <<  aTab[aK] << " " ;

     std::cout <<  "\n";
     ELISE_ASSERT(false,"Value is not in eligible set ");
}

class cAppliMalt 
{
     public :
         cAppliMalt(int argc,char ** argv);
         const std::string & Com() {return mCom;}
         const std::string & ComOA() {return mComOA;}

         int Exe();
     private :

          void ReadType(const std::string & aType);
          void InitDefValFromType();
          void ShowParam();

          bool        mModeHelp;
          std::string mStrType;
          eTypeMalt   mType;

          std::string mFullName;
          std::string mDir;
          cInterfChantierNameManipulateur * mICNM;
          const cInterfChantierNameManipulateur::tSet * mSetIm;
          int         mNbIm;
          std::string mIms;
          std::string mOri;

          int         mSzW;
          int         mNbMinIV;
          double      mZRegul;
          int         mUseMasqTA;
          int         mZoomFinal;
          int         mZoomInit;
          int         mExe;
          int         mNbEtapeQ;
          bool        mOrthoQ;
          bool        mOrthoF;
          double      mZPas;
          std::string mRep;
          bool        mRepIsAnam;
          std::string mCom;
          std::string mComOA;
          std::string mDirTA;
          bool        mPurge;
          bool        mMkFPC;
          bool        mDoMEC;
          bool        mUnAnam;
          bool        mOrthoInAnam;
          bool        mDoubleOrtho;
          double      mZincCalc;
          std::string mDirMEC;
          std::string mDirOrthoF;
          double       mDefCor;
          double       mCostTrans;
          int          mEtapeInit;
          bool         mAffineLast;
          std::string  mImMaster;
          double       mResolOrtho;

          std::string  mImMNT;
          std::string  mImOrtho;
          double       mZMoy;
          bool         mIsSperik;
};


int cAppliMalt::Exe()
{
  if (! mExe) return 0;
  int aRes = system_call(mCom.c_str());
  if ((aRes==0) && ( mComOA !=""))
     aRes = system_call(mComOA.c_str());
  ShowParam();
  return aRes;
}

cAppliMalt::cAppliMalt(int argc,char ** argv) :
    mICNM       (0),
    mNbMinIV    (3),
    mUseMasqTA  (1),
    mZoomInit   (-1),
    mExe        (1),
    mZPas       (0.4),
    mDirTA      ("TA"),
    mPurge      (true),
    mMkFPC      (true),
    mDoMEC      (true),
    mUnAnam      (true),
    mOrthoInAnam (false),
    mDoubleOrtho  (false),
    mZincCalc     (0.3),
    mDirMEC       ("MEC-Malt/"),
    mDirOrthoF    (""),
    mDefCor       (0.2),
    mCostTrans       (2.0),
    mEtapeInit    (1),
    mAffineLast   (true),
    mImMaster     (""),
    mResolOrtho   (1.0),
    mImMNT        (""),
    mImOrtho      (""),
    mIsSperik     (false)
{
  ELISE_ASSERT(argc >= 2,"Not enouh arg");

  ReadType(argv[1]);

  InitDefValFromType();

  



  std::string aMode;
  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(aMode,"Mode of correlation (must be in allowed enumerated values)")
                    << EAMC(mFullName,"Full Name (Dir+Pattern)")
                    << EAMC(mOri,"Orientation"),
        LArgMain()  << EAM(mImMaster,"Master",true," Master image must  exist iff Mode=GeomImage")
                    << EAM(mSzW,"SzW",true,"Correlation Window Size (1 means 3x3)")
                    << EAM(mZRegul,"Regul",true,"Regularization factor")
                    << EAM(mDirMEC,"DirMEC",true,"Subdirectory where the results will be stored")
                    << EAM(mDirOrthoF,"DirOF","Subdirectory for ortho (def in Ortho-${DirMEC}) ")
                    << EAM(mUseMasqTA,"UseTA",true,"Use TA as Masq when it exists (Def is true)")
                    << EAM(mZoomFinal,"ZoomF",true,"Final zoom, (Def 2 in ortho,1 in MNE)")
                    << EAM(mZoomInit,"ZoomI",true,"Initial Zoom, (Def depends on number of images)")
                    << EAM(mZPas,"ZPas",true,"Quantification step in equivalent pixel (def is 0.4)")
                    << EAM(mExe,"Exe",true,"Execute command (Def is true !!)")
                    << EAM(mRep,"Repere",true,"Local system of coordinat")
                    << EAM(mNbMinIV,"NbVI",true,"Number of Visible Image required (Def = 3)")
                    << EAM(mOrthoF,"HrOr",true,"Compute High Resolution Ortho")
                    << EAM(mOrthoQ,"LrOr",true,"Compute Low Resolution Ortho")
                    << EAM(mDirTA,"DirTA",true,"Directory  of TA (for mask)")
                    << EAM(mPurge,"Purge",true,"Purge the directory of Results before compute")
                    << EAM(mDoMEC,"DoMEC",true,"Do the Matching")
                    << EAM(mUnAnam,"UnAnam",true,"Compute the un-anamorphosed DTM and ortho (Def context dependant)")
                    << EAM(mDoubleOrtho,"2Ortho",true,"Do both anamorphosed ans un-anamorphosed ortho (when applyable) ")
                    << EAM(mZincCalc,"ZInc",true,"Incertitude on Z (in proportion of average depth, def=0.3) ")
                    << EAM(mDefCor,"DefCor",true,"Default Correlation in un correlated pixels (Def = 0.2) ")
                    << EAM(mCostTrans,"CostTrans",true,"Cost to change from correlation to uncorrelation (Def = 2.0) ")
                    << EAM(mEtapeInit,"Etape0",true,"First Step (Def=1) ")
                    << EAM(mAffineLast,"AffineLast",true,"Affine Last Etape with Step Z/2 (Def=true) ")
                    << EAM(mResolOrtho,"ResolOrtho",true,"Resolution of ortho, relatively to images (Def=1.0; 0.5 mean smaller images) ")
                    << EAM(mImMNT,"ImMNT",true,"Filter to select images used for matching (Def All, usable with ortho) ")
                    << EAM(mImOrtho,"ImOrtho",true,"Filter to select images used for ortho (Def All) ")
                    << EAM(mZMoy,"ZMoy",true,"Average value of Z")
                    << EAM(mIsSperik,"Spherik",true,"If true the surface for redressing are spheres")
  );

  if ((mImMaster!="") != (mType==eGeomImage))
  {
      std::cout << "Master Image =[" << mImMaster << "] , mode = " << mStrType << "\n";
      ELISE_ASSERT
      (
          false,
          "Incoherence : master image must exit iff mode==GeomImage"
      );
  }

  if (mEtapeInit!=1) 
     mPurge = false;
  MakeFileDirCompl(mDirMEC);
  if (mDirOrthoF=="")
     mDirOrthoF = "Ortho-" + mDirMEC;
  MakeFileDirCompl(mDirOrthoF);


  if (mModeHelp) 
     exit(-1);

  {
      int TabZF[3] ={1,2,4};
      VerifIn(mZoomFinal,TabZF,3,"ZoomFinal");
  }


  SplitDirAndFile(mDir,mIms,mFullName);
  mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
  mSetIm = mICNM->Get(mIms);
  mNbIm = mSetIm->size();
  ELISE_ASSERT(mNbIm>=2,"Not Enough image in Pattern");

  bool IsOrthoXCSte = false;
  mRepIsAnam = (mRep!="") && RepereIsAnam(mDir+mRep,IsOrthoXCSte);
  mUnAnam = mUnAnam && IsOrthoXCSte;

  if (mUnAnam)
  {
      mOrthoInAnam = mOrthoF;
      if (! mDoubleOrtho)
      {
          mOrthoF = false;
          mOrthoQ = false;
      }
  }

  if (mZoomInit!=-1)
  {
      int TabZI[4] ={128,64,32,16};
      VerifIn(mZoomInit,TabZI,4,"Zoom Init");
  }
  else
  {
       if (mNbIm > 1000) 
          mZoomInit = 128;
       else if (mNbIm > 100) 
          mZoomInit = 64;
       else if (mNbIm > 10 ) 
          mZoomInit = 32;
       else 
          mZoomInit = 32;
  }

  bool UseMTAOri = mUseMasqTA;

  mUseMasqTA =    UseMTAOri
               && ELISE_fp::exist_file(mDir+ELISE_CAR_DIR+ mDirTA +ELISE_CAR_DIR+"TA_LeChantier_Masq.tif");

  std::string FileMasqT = mUseMasqTA ? "MM-MasqTerrain.xml" : "EmptyXML.xml";

  if (mImMaster!="")
  {
    if (! EAMIsInit(&mDirMEC))
    {
        mDirMEC = "MM-Malt-Img-" + StdPrefix(mImMaster) +ELISE_CAR_DIR;
    }
    FileMasqT = "MM-MasqImage.xml";
    mUseMasqTA = UseMTAOri && ELISE_fp::exist_file(StdPrefix(mImMaster)+"_Masq.tif");
  }

// mZoomInit

  std::string aFileMM = "MM-Malt.xml";

  if (0) // (MPD_MM())
  {
       std::cout << "TTTTESSTTTTTT  MALT  !!!!!!!!\n";//   getchar();
       aFileMM = "Test-MM-Malt.xml";
       mPurge = false;
       mMkFPC = false;
  }


  mNbEtapeQ =    1   // Num premiere etape
              + round_ni(log2(mZoomInit/ mZoomFinal))  // Si aucune dupl
              + 1   //  Dulication de pas a la premiere
              + (mAffineLast ? 1 : 0)  ;  // Raffinement de pas;
 
  std::cout << 3+ log2(mZoomInit/ mZoomFinal)  << "\n";
  ShowParam();


  mPurge = mPurge &&  mDoMEC; 

  std::string  anArgCommuns =   std::string(" WorkDir=") + mDir
                             +  std::string(" +ImPat=") + QUOTE(mIms)
                             +  std::string(" +DirMEC=") + mDirMEC
                             +  std::string(" +ZoomFinal=") + ToString(mZoomFinal)
                             +  std::string(" +Ori=") + mOri
                             +  std::string(" +ResolRelOrhto=") + ToString(1/(mResolOrtho*mZoomFinal))
                             +  std::string(" +DirTA=") + mDirTA
                 ;


  std::string aNameGeom = (mImMaster=="") ? "eGeomMNTEuclid" : (mIsSperik? "eGeomMNTFaisceauPrChSpherik" :"eGeomMNTFaisceauIm1PrCh_Px1D");

  mCom =     MMDir() +"bin"+ELISE_CAR_DIR+"MICMAC "
                      +  MMDir() +"include"+ELISE_CAR_DIR+"XML_MicMac"+ELISE_CAR_DIR+aFileMM // MM-Malt.xml
                      + anArgCommuns

/*
                      +  std::string(" +DirTA=") + mDirTA
                      +  std::string(" WorkDir=") + mDir
                      +  std::string(" +ImPat=") + QUOTE(mIms)
                      +  std::string(" +DirMEC=") + mDirMEC
                      +  std::string(" +ZoomFinal=") + ToString(mZoomFinal)
                      +  std::string(" +Ori=") + mOri
                      +  std::string(" +ResolRelOrhto=") + ToString(1.0/mZoomFinal)
                      +  std::string(" +DirOrthoF=") + mDirOrthoF
*/


                      +  std::string(" +DirOrthoF=") + mDirOrthoF
                      +  std::string(" +ZRegul=") + ToString(mZRegul)
                      +  std::string(" +SzW=") + ToString(mSzW)
                      +  std::string(" +ZoomInit=") + ToString(mZoomInit)
                      +  std::string(" +FileMasqT=") + FileMasqT

                      +  std::string(" +ZPas=") + ToString(mZPas)
                      +  std::string(" +DbleZPas=") + ToString(mZPas*2)
                      +  std::string(" +DemiZPas=") + ToString(mZPas/2)
                      +  std::string(" +NbMinIV=") + ToString(mNbMinIV)

                      + std::string(" +FileOthoF=") + (mOrthoF ? "MM-Malt-OrthoFinal.xml" : "EmptyXML.xml")
                      + std::string(" +FileOthoQ=") + (mOrthoQ ? "MM-Malt-OrthoQuick.xml" : "EmptyXML.xml")

                      + std::string(" +FileUnAnam=") + (mUnAnam ? "MM-Malt-UnAnam.xml" : "EmptyXML.xml")

                      + std::string(" +Purge=") + (mPurge ? "true" : "false")
                      + std::string(" +MkFPC=") + (mMkFPC ? "true" : "false")
                      + std::string(" +DoMEC=") + (mDoMEC ? "true" : "false")
                      + std::string(" +ZIncCalc=") + ToString(mZincCalc)
                      + std::string(" +NbEtapeQuant=") + ToString(mNbEtapeQ)
                      + std::string(" +DefCor=") + ToString(mDefCor)
                      + std::string(" +mCostTrans=") + ToString(mCostTrans)
                      + std::string(" +Geom=") + aNameGeom
                  ;



  if (mImMaster != "")
  {
     mCom =  mCom
           + std::string(" +ImageMaster=") + mImMaster
           + std::string(" +ImageMasterSsPost=") + StdPrefix(mImMaster)
           + std::string(" +FileIm1=") + "MM-ImageMaster.xml" 
           + std::string(" +ZIncIsProp=") + "false"
           + std::string(" +FullIm1=") + "true"
           + std::string(" +PasInPixel=") + "false"
          ;
  }

  if (mRep!="")
  {
      if (!mRepIsAnam)
      {
          mCom = mCom +  std::string(" +Repere=") + mRep;
      }
      else
      {
        mCom =    mCom
               +  std::string(" +FileAnam=") + "MM-Anam.xml"
               +  std::string(" +ParamAnam=") + mRep;
      }
  }

  if (!mAffineLast)
  {
      mCom = mCom +  " +FileZ1Raff=EmptyXML.xml";
  }

  if (mEtapeInit!=1)
  {
      mCom = mCom + " FirstEtapeMEC="+ToString(mEtapeInit);
  }


  if (mZoomFinal==1)
  {
       mCom = mCom + std::string(" +FileZ2PC=MM-Zoom2-PC.xml") ;
  }
  else if (mZoomFinal==4)
  {
       mCom = mCom + std::string(" +FileZ4PC=EmptyXML.xml") ;
  }

  if (mZoomInit >= 128)
     mCom = mCom + std::string(" +FileZ64=MM-Zoom64.xml");
  if (mZoomInit >=64)
     mCom = mCom + std::string(" +FileZ32=MM-Zoom32.xml");
  if (mZoomInit <= 16)
     mCom = mCom + std::string(" +FileZ16==EmptyXML.xml");

  if (EAMIsInit(&mZMoy))
  {
        mCom = mCom + " +FileZMoy=File-ZMoy.xml"
                    + " +ZMoy=" + ToString(mZMoy);
  }

  
               
  std::cout << mCom << "\n";
  // cInZRegulterfChantierNameManipulateur * aCINM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

  if (mImMNT !="") mCom   =  mCom + std::string(" +ImMNT=")   + mImMNT;
  if (mImOrtho !="") mCom =  mCom + std::string(" +ImOrtho=") + mImOrtho;
  if (mOrthoInAnam)
  {
       std::string aFileOAM  = "MM-Malt-OrthoAnamOnly.xml";

	   mComOA =  MMDir() +"bin"+ELISE_CAR_DIR+"MICMAC "
               + MMDir() +"include"+ELISE_CAR_DIR+"XML_MicMac"+ELISE_CAR_DIR+aFileOAM // MM-Malt.xml 
               + anArgCommuns;

       mComOA =        mComOA 
                    +  std::string(" +Repere=") + mRep
                    +  std::string(" +DirOrthoF=") +  "Ortho-UnAnam-" + mDirMEC
                 ;

       if (mImMNT !="") mComOA   =  mComOA + std::string(" +ImMNT=")   + mImMNT;
       if (mImOrtho !="") mComOA =  mComOA + std::string(" +ImOrtho=") + mImOrtho;
       std::cout << "\n\n" << mComOA << "\n";
  }
}

     // mDirOrthoF = "Ortho-" + mDirMEC;

void cAppliMalt::InitDefValFromType()
{
  switch (mType)
  {
        case eOrtho :
             mSzW = 2;
             mZRegul = 0.05;
             mZoomFinal = 2;
             mOrthoF = true;
             mOrthoQ = true;
             mAffineLast = (mZoomFinal != 1);
        break;

        case eUrbanMNE :
             mSzW = 1;
             mZRegul = 0.02;
             mZoomFinal = 1;
             mOrthoF = false;
             mOrthoQ = false;
             mAffineLast = false;
        break;

        case eGeomImage :
             mSzW = 1;
             mZRegul = 0.02;
             mZoomFinal = 1;
             mOrthoF = false;
             mOrthoQ = false;
             mAffineLast = false;
        break;

        case eNbTypesMNE :
        break;
  };

}

void  cAppliMalt::ShowParam()
{
  std::cout << "============= PARAMS ========== \n";

  std::cout <<  " -  SzWindow " << mSzW << "  (i.e. : " << 1+2*mSzW << "x" << 1+2*mSzW << ")\n";
  std::cout <<  " -  Regul " <<  mZRegul   << "\n";
  std::cout <<  " -  Final Zoom " <<  mZoomFinal   << "\n";
  std::cout <<  " -  Initial Zoom " <<  mZoomInit   << "\n";
  std::cout <<  " -  Use TA as Mask  " <<  (mUseMasqTA ? " Yes" : " No")   << "\n";
  std::cout <<  " -  Z Step : " <<  mZPas   << "\n";
  if (mRep!="")
     std::cout <<  " -  Repere  : " <<  mRep   << "\n";
  std::cout <<  " -  Nb Min Visible Images : " <<  mNbMinIV   << "\n";
  std::cout << "================================ \n";

}



void cAppliMalt::ReadType(const std::string & aType)
{
    mStrType = aType;
    StdReadEnum(mModeHelp,mType,mStrType,eNbTypesMNE);
}



int main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);
   cAppliMalt anAppli(argc,argv);

   return anAppli.Exe();
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
