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


/*
*/

#include "StdAfx.h"



/************************************************************/
/*                                                          */
/*                   cParamRechCible                        */
/*                                                          */
/************************************************************/

static const std::string  NoDefStr = "x@@%?&!88!$\243";


cParamRechCible::cParamRechCible
(
     bool isStepInit,
     INT  aSzW,
     REAL aDistConf,
     REAL anEllipseConf
) :
   mStepInit    (isStepInit),
   mSzW         (aSzW),
   mDistConf    (aDistConf),
   mEllipseConf (anEllipseConf)
{
}


/************************************************************/
/*                                                          */
/*                   cParamEtal                             */
/*                                                          */
/************************************************************/

std::string cParamEtal::NameTiff(const std::string Im) const
{
     return mDirectory + NameTiffSsDir(Im);
}

std::string cParamEtal::NameTiffSsDir(const std::string Im) const
{
     return  mPrefixe + Im + mPostfixe;
}

Pt2di                cParamEtal::SzIm()      const {return mSzIm;}
const std::string & cParamEtal::Directory()  const {return mDirectory;}
const std::string & cParamEtal::NameFile()  const {return mNameFile;}
REAL                cParamEtal::FocaleInit() const {return mFocaleInit;}
const cParamRechCible &  cParamEtal::ParamRechInit() const {return mPRCInit;}
const cParamRechCible &  cParamEtal::ParamRechDRad() const {return mPRCDrad;}
const std::string & cParamEtal::PrefixeTiff()  const {return mPrefixe;}
INT cParamEtal::Zoom() const {return mZoom;}
const std::vector<std::string>  & cParamEtal::ImagesInit() const {return mImagesInit;}
const std::vector<std::string>  & cParamEtal::AllImagesCibles() const {return mImages;}
const std::vector<std::string>  & cParamEtal::CamRattachees() const {return mCamRattachees;}

std::string cParamEtal::NameCible3DPolygone() const
{return mPolygoneDirectory+mNameCible3DPolygone;}
std::string cParamEtal::NameImPolygone() const
{return mPolygoneDirectory+mNameImPolygone;}
std::string cParamEtal::NamePointePolygone() const
{return mPolygoneDirectory+mNamePointePolygone;}

const std::string & cParamEtal::PointeInitIm() {return mPointeInitIm;}


const std::string & cParamEtal::NameCamera()  const {return mNameCamera;}
const std::string & cParamEtal::NameFileRatt() const
{
     ELISE_ASSERT(mNameFileRatt != NoDefStr,"No Rattachement for Camera");
     return mNameFileRatt;
}
bool cParamEtal::HasFileRat() const {return mNameFileRatt != NoDefStr;}
const std::string & cParamEtal::NamePosRatt() const
{
     ELISE_ASSERT(mNamePosRattach != NoDefStr,"No Rattachement for Camera");
     return mNamePosRattach;
}

REAL cParamEtal::SeuilCorrel() const {return mSeuilCorrel;}


REAL  cParamEtal::TaillePixelExportLeica() const
{
   return mTaillePixelExportLeica;
}

bool cParamEtal::ModeMTD() const
{
   return mTaillePixelExportLeica > 0;
}

bool   cParamEtal::ModeC2M() const
{
   return mModeC2M != 0;
}
double cParamEtal::DefLarg() const {return mDefLarg;}

Tiff_Im cParamEtal::TiffFile(const std::string Im) const
{
  return  Tiff_Im::UnivConvStd(NameTiff(Im));
}


std::string cParamEtal::ANameFileExistant()
{
   std::string aName = NameTiff(mImagesInit[0]) ;
   if (ELISE_fp::exist_file(aName))
      return aName;
/*
*/
   std::cout << "TEST THOM VIDE " <<  aName << "\n";
   ELISE_ASSERT(false,"Fichier thom vide : mecanisme obsolete");
   std::string aNameVide = mDirectory + "FichierThomVide";
   MakeFileThomVide
   (
      aNameVide,
      NameTiff(aName)
   );
   return aNameVide;
}

eUseCibleInit  Str2UCI(const std::string & aName)
{
   if (aName == "Jamais")
      return eUCI_Jamais;
   if (aName == "Toujours")
      return eUCI_Toujours;
   if (aName == "OnEchec")
      return eUCI_OnEchec;
   if (aName == "Only")
      return eUCI_Only;


    std::cout << "Val=[" << aName <<"]\n";
    ELISE_ASSERT(false,"Bad Value for UsePointeManuel");
   return eUCI_OnEchec;
}


void cParamEtal::FilterImage(std::vector<std::string> &  mVIM)
{
   if (! Im2Select().IsInit())
      return ;
   if (! Im2Select().Val().UseIt().Val())
      return ;
   mVIM = Im2Select().Val().Id();
}

bool  cParamEtal::DoSift() const
{
   return  mAutomSift  != 0;
}

bool cParamEtal::CalibSpecifLoemi() const
{
   return ( mCalibSpecifLoemi!=0 );
}

bool  cParamEtal::DoSift(const std::string & aName) const
{
   return     (mAutomSift  != 0)
           && (mAutomSift->Match(aName));
}

const std::string  &  cParamEtal::KeyExportOri() const
{
   return mKeyExportOri;
}
const std::string  &  cParamEtal::KeySetOri() const
{
   return mKeySetOri;
}
const int  &  cParamEtal::DoGrid() const
{
   return mDoGrid;
}

const std::string & cParamEtal::PatternGlob() const
{
   return mPatternGlob;
}




void cParamEtal::InitFromFile(const std::string & aNameFile)
{
     static std::string  aPatternCamV2 = ".*_.x[0-9]{2,3}_[0-9]{4,5}($|\\.tif|_8Bits\\.tif)";
     static cElRegex   anAutomCamV2(aPatternCamV2,10);


     mZoom = -1;
     mSzIm = Pt2di(-1,-1);
     mSeuilCorrel = 0.9;
     mNameCible3DPolygone = "IGNPoly";
     mNameImPolygone = "poly86_1.tif";
     mNamePointePolygone = "IGNPointePoly";

     mNameFileRatt   = NoDefStr;
     mNamePosRattach = NoDefStr;
     mEgelsConvImage = 0;

     mCibDirU = Pt3dr(1,0,0);
     mCibDirV = Pt3dr(0,0,1);

     mCDistLibre = -1;
     mDegDist    = 3;
     mMakeImageCibles = 0;
     mSeuilCoupure = 1e10;
     mInvYPointe = 0;
     mRabExportGrid = 100;


     std::string aTypeResol ="L2";

     std::string aUCI_Init = "OnEchec";
     std::string aUCI_DRad = "OnEchec";

     mNamePolygGenImageSaisie ="";
     mNameImageGenImageSaisie ="";
     mZoomGenImageSaisie       =2.0;

     mStepGridXML = 10.0;
     mXMLAutonome = 0;
     mNbPMaxOrient =1000;
     mTaillePixelExportLeica = -1;
     mModeC2M = 0;
     mByProcess = 0;
     mCalledByItsef = 0;
     mPointeInitIm = "InitIm";

     mAutomSift = 0;
     std::string aPatternSift ="";

     mDefLarg = 0.7;

     std::string aNameCompl = "";
     std::string aPatternIm="";

     mKeyExportOri = "";
     mKeySetOri = "";
     mCalibSpecifLoemi = -1;
     mDoGrid = -1;

     int  aMkFileVide = 0;

     mSeuilRejetEcart = 1.0;
     mSeuilRejetAbs   = 10.0;
     mSeuilPonder     = 0.4;

     LArgMain anArg;
     StdInitArgsFromFile
     (
          aNameFile,
          anArg
              <<   EAM(mPolygoneDirectory,"PolygoneDirectory",false)
              <<   EAM(mDirectory,"Directory",false)
              <<   EAM(mImages,"ListeImages",true)
              <<   EAM(aPatternIm,"PatternImage",true)
              <<   EAM(aPatternSift,"PatternSift",true)
          <<   EAM(mPrefixe,"PrefixeNomImage",false)
          <<   EAM(mImagesInit,"ListeImagesInit",false)
          <<   EAM(mFocaleInit,"FocaleInit",false)
          <<   EAM(mNameCamera,"NameCamera",false)

          <<   EAM(mKeyExportOri,"KeyExportOri",true)
          <<   EAM(mKeySetOri,"KeySetOri",true)

          <<   EAM(mPostfixe,"PostfixeNomImage",true)
          <<   EAM(mNameFileRatt,"EtalonnageRattachement",true)
          <<   EAM(mNamePosRattach,"PositionRattachement",true)
          <<   EAM(mCamRattachees,"CameraRattachees",true)
          <<   EAM(mZoom,"Zoom",true)
          <<   EAM(mSzIm,"SzIm",true)

          <<   EAM(mNameCible3DPolygone,"NameCible3DPolygone",true)
          <<   EAM(mNamePointePolygone,"NamePointePolygone",true)
          <<   EAM(mNameImPolygone,"NameImPolygone",true)

          <<   EAM(mPRCInit.mSzW,"FenetreInit",true)
          <<   EAM(mPRCInit.mDistConf,"DistConfInit",true)
          <<   EAM(mPRCInit.mEllipseConf,"IncFormeInit",true)
          <<   EAM(aUCI_Init,"UsePointeManuelInit",true)

          <<   EAM(mPRCDrad.mSzW,"FenetreDrad",true)
          <<   EAM(mPRCDrad.mDistConf,"DistConfDrad",true)
          <<   EAM(mPRCDrad.mEllipseConf,"IncFormeDrad",true)
          <<   EAM(aUCI_DRad,"UsePointeManuelDRad",true)

              <<   EAM(mSeuilCorrel,"SeuilCorrel",true)
          <<   EAM(mEgelsConvImage,"EgelsConvImage",true)
          <<   EAM(mCibDirU,"CibDirU",true)
          <<   EAM(mCibDirV,"CibDirV",true)
          <<   EAM(mCiblesRejetees,"CiblesRejetees",true)
          <<   EAM(mCDistLibre,"CDistLibre",true)
          <<   EAM(mDegDist,"DegDist",true)
              <<   EAM(mSeuilCoupure,"SeuilCoupure",true)
              <<   EAM(mInvYPointe,"InvYPointe",true)
              <<   EAM(mRabExportGrid,"RabExportGrid",true)
              <<   EAM(mCalibSpecifLoemi,"IsLoemi",true)
              <<   EAM(mDoGrid,"DoGrid",true)
              <<   EAM(aTypeResol,"TypeResol",true)
          <<   EAM(mNamePolygGenImageSaisie,"NamePolygGenImageSaisie",true)
          <<   EAM(mNameImageGenImageSaisie,"NameImageGenImageSaisie",true)
          <<   EAM(mZoomGenImageSaisie,"ZoomGenImageSaisie",true)

              <<   EAM(mXMLAutonome,"XMLAutonome",true)
              <<   EAM(mStepGridXML,"StepGridXML",true)
              <<   EAM(mNbPMaxOrient,"NbPMaxOrient",true)
              <<   EAM(mTaillePixelExportLeica,"TaillePixelExportLeica",true)
              <<   EAM(mModeC2M,"ModeC2M",true)
              <<   EAM(mByProcess,"ByP",true)
              <<   EAM(mPointeInitIm,"Pointe",true)
              <<   EAM(mDefLarg,"DefLarg",true)
              <<   EAM(aNameCompl,"XML-Add",true)
              <<   EAM(aMkFileVide,"MkFileVide",true)
              <<   EAM(mSeuilRejetEcart,"SeuilRejet",true)
              <<   EAM(mSeuilRejetAbs,"SeuilRejetAbs",true)
              <<   EAM(mSeuilPonder,"SeuilPonder",true)
     );

     mDirectory = StdWorkdDir(mDirectory,aNameFile);
     mNameFile = NameWithoutDir(aNameFile);

     ELISE_ASSERT
     (
           (!mImages.empty()) || (aPatternIm!=""),
           "Ni images ni pattern"
     );

     if (aPatternIm!="")
     {
        mPatternGlob = mPrefixe+ "("+aPatternIm+")"+mPostfixe;

        std::string aPatSubst =  "$1";
        std::list<std::string> aLP=RegexListFileMatch(mDirectory,mPatternGlob,1,false);
        cElRegex anAutom(mPatternGlob,10);
        for
        (
            std::list<std::string>::const_iterator itS=aLP.begin();
            itS!= aLP.end();
            itS++
        )
        {
            std::string aNewS = MatchAndReplace(anAutom,*itS,aPatSubst);
            mImages.push_back(aNewS);
            // std::cout << aNewS << "\n";
        }
       // getchar();
     }
     else
     {
        mPatternGlob = mPrefixe+ "(";
        for (int aK=0 ; aK<int(mImages.size()) ; aK++)
        {
           if (aK!=0 )
               mPatternGlob = mPatternGlob + "|";
           mPatternGlob = mPatternGlob + mImages[aK];
           if (aMkFileVide)
           {
              std::string  aName = mDirectory + mPrefixe+mImages[aK] + mPostfixe;
              if (! ELISE_fp::exist_file(aName))
              {
                  FILE * aFP = ElFopen(aName.c_str(),"w");
                  ElFclose(aFP);
              }
           }
        }

        mPatternGlob = mPatternGlob+")" + mPostfixe;
     }


/// std::cout << "TTT2="<< anAutomCamV2.Match("img_0582_MpDcraw8B_GR.tif") << "\n";
/// std::cout << "TTT2="<< anAutomCamV2.Match("img_0582_MpDcraw8B_GR.tif") << "\n";
/// std::cout << "mCalibSpecifLoemi " << mCalibSpecifLoemi << "\n";
/// std::cout << "aPatternCamV2["<< aPatternCamV2 << "]\n";

     if (mCalibSpecifLoemi<0)
     {
        mCalibSpecifLoemi=(! mImages.empty())  && anAutomCamV2.Match(mPrefixe+mImages[0]+mPostfixe);
     }


//  std::cout << "mCalibSpecifLoemi " << mCalibSpecifLoemi  << " " <<  mPrefixe+mImages[0]+mPostfixe << "\n";

    if (mCalibSpecifLoemi)
    {
        std::string aNameAuto = "Syst([0-9]{1,3}_[0-9]{2,4}_[0-9]{1,2}_[0-9]{2,3}_[rvbilmnop])";
        cElRegex anAutom(aNameAuto,10);
        if (! anAutom.Match(mNameCamera))
        {
            std::string aNameAuto_NPOL = "Syst(2_09_80_NP_.*)";
            cElRegex anAuto_NPOL(aNameAuto_NPOL,10);
            if (anAuto_NPOL.Match(mNameCamera))
            {
                mNameCamera = MatchAndReplace(anAuto_NPOL,mNameCamera,"$1");
            }
            else
            {
               std::cout << "Nom de camera " << mNameCamera << "\n";
               std::cout << "Automate de specif : " << aNameAuto << "\n";
               ELISE_ASSERT(false," Nom de camera incorrecte");
            }
        }
        else
        {
           mNameCamera = MatchAndReplace(anAutom,mNameCamera,"$1");
        }
     }


     if (mKeyExportOri=="")
          mKeyExportOri =   mCalibSpecifLoemi ?
                            "Key-Assoc-Im2AppuiOri-Polygone-CamV2" :
                            "Key-Assoc-Im2AppuiOri-Polygone-Basic" ;
     if (mKeySetOri=="")
         mKeySetOri = mCalibSpecifLoemi                 ?
                      "Key-Set-AppuiOri-Polygone-CamV2" :
                      "Key-Set-AppuiOri-Polygone-Basic" ;

      if (mDoGrid==-1)
        mDoGrid= mCalibSpecifLoemi;


     if (aPatternSift!="")
        mAutomSift = new cElRegex(aPatternSift,10);




     ELISE_ASSERT
     (
        !mModeC2M,
    "Mode C2M Deconseille"
     );

     if (aNameCompl != "")
     {
        (cComplParamEtalPoly &) (*this) =
                  StdGetObjFromFile<cComplParamEtalPoly>
              (
                         mDirectory+aNameCompl,
                         StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                          "ComplParamEtalPoly",
                          "ComplParamEtalPoly"
              );
     }

     mPRCInit.mUseCI = Str2UCI(aUCI_Init);
     mPRCDrad.mUseCI = Str2UCI(aUCI_DRad);

     if (aTypeResol=="L2")
        mTypeSysResolve = cNameSpaceEqF::eSysPlein;
     else if (aTypeResol=="L1")
        mTypeSysResolve = cNameSpaceEqF::eSysL1Barrodale;
     else
     {
         ELISE_ASSERT
     (
         false,
         "Unknown Type Resol"
     );
     }

     FilterImage(mImagesInit);
     FilterImage(mImages);


     ELISE_ASSERT(mImagesInit.size()!=0,"Empty Image Init");


     if ((mSzIm.x<=0) || (mSzIm.y<=0))
     {
        std::string aName = ANameFileExistant();
        Tiff_Im aTif = Tiff_Im::UnivConvStd(aName);
        mSzIm = aTif.sz();
     }

     std::sort(mCiblesRejetees.begin(),mCiblesRejetees.end());

     cout    <<  "SzIm = " << mSzIm
         << " FenetreInit " << mPRCInit.mSzW << "\n";


}

std::string  cParamEtal::NamePolygGenImageSaisie () const
{
  return mPolygoneDirectory + mNamePolygGenImageSaisie;
}

std::string cParamEtal::FullNameImageGenImageSaisie () const
{
  return mPolygoneDirectory + mNameImageGenImageSaisie +".txt";
}

std::string cParamEtal::ShortNameImageGenImageSaisie () const
{
  return mNameImageGenImageSaisie ;
}
double  cParamEtal::ZoomGenImageSaisie () const
{
   return mZoomGenImageSaisie;
}

bool  cParamEtal::HasGenImageSaisie () const
{
   return mNamePolygGenImageSaisie != "";
}

bool cParamEtal::CalledByItsef() const
{
   return mCalledByItsef != 0;
}
int cParamEtal::ByProcess() const
{
   return mByProcess;
}


cNameSpaceEqF::eTypeSysResol  cParamEtal::TypeSysResolve() const
{
   return mTypeSysResolve;
}

Pt2di cParamEtal::RabExportGrid() const
{
   return Pt2di(mRabExportGrid,mRabExportGrid);
}

bool cParamEtal::InvYPointe() const
{
   return ( mInvYPointe!=0 );
}

INT  cParamEtal::DegDist() const {return mDegDist;}

bool cParamEtal::CDistLibre(bool aDef) const
{
     if (mCDistLibre < 0)  return aDef;
     return (mCDistLibre != 0) ;
}

REAL cParamEtal::SeuilCoupure() const
{
  return mSeuilCoupure;
}

bool cParamEtal::AcceptCible(INT anInd) const
{
    return ! binary_search(mCiblesRejetees.begin(),mCiblesRejetees.end(),anInd);
}

const std::vector<INT>  &  cParamEtal::CiblesRejetees() const
{
   return mCiblesRejetees;
}

bool cParamEtal::MakeImagesCibles() const
{
    return mMakeImageCibles != 0;
}

cParamEtal::cParamEtal(int argc,char ** argv) :
    mPRCInit (true,256,100.0,0.2),
    mPRCDrad (false,64,10.0,0.1)
{
     mOrderComp = "PR5";
     mCibleDeTest = -1;
     mImDeTest    = "";
     ELISE_ASSERT(argc >=2,"Not Enough Arg in ParamAnag");
     InitFromFile(argv[1]);
     // LArgMain anArgObl;
     LArgMain anArgFac;
     std::string  UnUsed;

// std::cout << "cParamEtal " <<  argc << "\n";

     ElInitArgMain
     (
         // argc-1,argv+1,
         // anArgObl  ,
         argc,argv,
         LArgMain()  <<   EAM(UnUsed),
         anArgFac   << EAM(mCibleDeTest,"Cible",true)
                << EAM(mImDeTest,"Im",true)
                << EAM(mZoom,"Zoom",true)
            << EAM(mMakeImageCibles,"MakeIm",true)
                    << EAM(mPointeInitIm,"Pointe",true)
                    << EAM(mByProcess,"ByP",true)
                    << EAM(mCalledByItsef,"CalledByItsef",true)
                    << EAM(mOrderComp,"OC",true,"Order Compens, allow value in {P,R,5}")
     );

     // std::cout << "XXXXXXXx   " << mPointeInitIm << "\n";
     // getchar();
}




cParamEtal cParamEtal::FromStr(const std::string & aName)
{
    char * tab[2];
    tab[0] = 0;
    tab[1] = const_cast<char *>(aName.c_str());
    return cParamEtal(2,tab);
}

cParamEtal cParamEtal::ParamRatt()
{
    return FromStr(NameFileRatt());
}

INT cParamEtal::EgelsConvImage()
{
    return mEgelsConvImage;
}

Pt3dr cParamEtal::CibDirU() const {return mCibDirU;}
Pt3dr cParamEtal::CibDirV() const {return mCibDirV;}
INT                   cParamEtal::CibleDeTest () const {return mCibleDeTest;}
const std::string  &  cParamEtal::ImDeTest    () const {return mImDeTest;}

double cParamEtal::StepGridXML() const
{
   return mStepGridXML;
}

bool cParamEtal::XMLAutonome() const
{
   return mXMLAutonome != 0;
}


int cParamEtal::NbPMaxOrient() const
{
   return mNbPMaxOrient ;
}


double cParamEtal::SeuilRejetEcart() const { return mSeuilRejetEcart; }
double cParamEtal::SeuilRejetAbs()   const { return mSeuilRejetAbs; }
double cParamEtal::SeuilPonder()     const { return mSeuilPonder; }

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
