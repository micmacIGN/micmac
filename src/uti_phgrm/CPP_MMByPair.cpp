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
const std::string TheDIRMergeEPI(){return  "MTD-Image-";}

cElemAppliSetFile & cAppliWithSetImage::EASF() {return mEASF;}
const cElemAppliSetFile & cAppliWithSetImage::EASF() const {return mEASF;}

const std::string cAppliWithSetImage::TheMMByPairNameCAWSI =  "MMByPairCAWSI.xml";
const std::string cAppliWithSetImage::TheMMByPairNameFiles =  "MMByPairFiles.xml";

extern double DynCptrFusDepthMap;

std::string PatFileOfImSec(const std::string & anOri)
{
   return  "NKS-Set-OfFile@Ori-" + anOri + "/FileImSel.xml";
}
std::string DirAndPatFileOfImSec(const std::string & aDir,const std::string & anOri)
{
   return aDir + "%" + PatFileOfImSec(anOri);
}

std::string DirAndPatFileMMByP(const std::string & aDir)
{
   return aDir + "%NKS-Set-OfFile@" + cAppliWithSetImage::TheMMByPairNameFiles;
}




bool IsMacType(eTypeMMByP aType)
{
     return  (aType==eBigMac) || (aType==eMicMac) || (aType==eQuickMac);
}

cPatOfName::cPatOfName() :
    mPat ("\"(") ,
    mNb (0)
{
}
std::string cPatOfName::Pattern() const
{
   return mPat + ")\"";
}

void cPatOfName::AddName(const std::string & aName)
{
   if (mNb) mPat += "|";
   mPat += aName;
   mNb++;
}


class cAppliClipChantier : public cAppliWithSetImage
{
    public :
        cAppliClipChantier(int argc,char ** argv);

        std::string  mNameMasterIm;
        Box2di       mBox;
        tSomAWSI *   mMasterIm;
};


class cAppliMMByPair : public cAppliWithSetImage
{
    public :
      cAppliMMByPair(int argc,char ** argv);
      int Exe();
    private :
      std::string DirMTDImage(const tSomAWSI &) const;

      void Inspect();
      bool InspectMTD(tArcAWSI & anArc,const std::string & aName );
      bool InspectMTD(tArcAWSI & anArc);
      bool InspectMTD_REC(tArcAWSI & anArc);


      void DoMDTGround();
      void DoMDTRIE(bool TiePM0);
      void DoMDT();

      void DoCorrelAndBasculeStd();
      void DoCorrelEpip();
      void DoReechantEpipInv();
      std::string MatchEpipOnePair(tArcAWSI & anArc,bool & ToDo,bool & Done,bool & Begun);
      void DoFusion();
      void DoFusionGround();
      void DoFusionEpip();

      std::string mDo;
      int mZoom0;
      int mZoomF;
      bool mParalMMIndiv;
      std::string mFilePair;
      bool mDelaunay;
      // Avant mAddMMImSec  ; maintenant on separe l'execution de AperoImSec (mRunAperoImSec) de l'ajout des arc ; car parfois
      // meme si on ne veut pas rajouter les arcs (par ex parce que controle explicite des paires par FilePair) on a besoin
      // des autres donnees generees
      bool mRunAperoImSec;
      bool mAddCpleImSec;
      bool mAddCpleLine;
      int mDiffInStrip;
      bool mStripIsFirt;
      std::string  mMasterImages;
      std::string  mPairByStrip;
      std::string  mDirBasc;
      int          mNbStep;
      double       mIntIncert;
      bool         mSkipCorDone;
      eTypeMMByP   mType;
      bool         mMacType;
      std::string  mStrType;
      bool         mByMM1P;
      // bool         mByEpi;
      Box2di       mBoxOfImage;
      std::string  mImageOfBox;
      std::string  mStrQualOr;
      eTypeQuality mQualOr;
      bool         mHasVeget;
      bool         mSkyBackGround;
      bool         mDoOMF;
      bool         mRIEInParal;  // Pour debuguer en l'inhibant,
      bool         mRIE2Do;      // Do Reech Inv Epip
      bool         mExeRIE;      // Do Reech Inv Epip
      bool         mDoTiePM0;      // Do model initial wih MMTieP ..
      int          mTimes;
      bool         mDebugCreatE;
      bool         mDebugMMByP;
      bool         mPurge;
      bool         mUseGpu;
      double       mDefCor;
      double       mZReg;
      bool	   mExpTxt;
      bool	   mExpImSec;
      bool mSuprImNoMasq;
      std::string mPIMsDirName;
      std::string mSetHom;
      double mTetaOpt;
};

/*****************************************************************/
/*                                                               */
/*                            ::                                 */
/*                                                               */
/*****************************************************************/

int TiffDev_main(int argc,char ** argv)
{
    std::string aNameFile;
    int aNbChan = -1 ;
    bool B16 = true;
    bool ExigNoCompr = false;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameFile,"File Name", eSAM_IsExistFile),
        LArgMain()  << EAM(aNbChan,"NbChan",true,"Nb Channel")
                    << EAM(B16,"16B",true,"Keep in 16 Bits if possible")
                    << EAM(ExigNoCompr,"ENC",true,"Exig No Compr")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    Tiff_Im::StdConvGen(aNameFile,aNbChan,B16,ExigNoCompr);

    return 0;
}

void Paral_Tiff_Dev
    (
         const std::string & aDir,
         const std::vector<std::string> & aLFile,
         int                            aNbChan,
         bool                           Cons16B
    )
{
    std::list<std::string> aLCom;
    for (std::vector<std::string>::const_iterator itS=aLFile.begin(); itS!=aLFile.end(); itS++)
    {
         std::string aCom =    MMBinFile(MM3DStr) + " TifDev " + aDir+*itS
                              + " NbChan=" + ToString(aNbChan)
                              + " 16B=" + ToString(Cons16B)
                            ;
         aLCom.push_back(aCom);
         // std::cout << aCom << "\n";
    }

    cEl_GPAO::DoComInParal(aLCom);
}

std::string PatternOfVois(tSomAWSI & aSom ,bool IncludeThis)
{
   cPatOfName aPON;
   if (IncludeThis) aPON.AddName(aSom.attr().mIma->mNameIm);

   cSubGrAWSI aSubGrAll;
   for (tItAAWSI itA=aSom.begin(aSubGrAll) ; itA.go_on() ; itA++)
       aPON.AddName((*itA).s2().attr().mIma->mNameIm);

/*
cImaMM*>::const_iterator itIM=mVois.begin() ; itIM!=mVois.end() ; itIM++)
    ;
   for (std::list<cImaMM*>::const_iterator itIM=mVois.begin() ; itIM!=mVois.end() ; itIM++)
       aPON.AddName((*itIM)->mNameIm);
*/

   return aPON.Pattern();
}

/*****************************************************************/
/*                                                               */
/*          cAttrSomAWSI , cAttrArcAWSI                          */
/*                                                               */
/*****************************************************************/

cAttrSomAWSI::cAttrSomAWSI() :
  mIma (0)
{
}
cAttrSomAWSI::cAttrSomAWSI(cImaMM* anIma,int aNumGlob,int aNumAccepted) :
  mIma         (anIma),
  mNumGlob     (aNumGlob),
  mNumAccepted (aNumAccepted)
{
}
cAttrArcAWSI::cAttrArcAWSI() :
  mCpleE (0)
{
}
cAttrArcAWSI::cAttrArcAWSI(cCpleEpip * aCpleE) :
  mCpleE (aCpleE)
{
}


std::string NameImage(tArcAWSI & anArc,bool Im1,bool ByEpi)
{
    tSomAWSI & aS = Im1 ? anArc.s1() : anArc.s2();
    std::string aRes = aS.attr().mIma->mNameIm;
    if (ByEpi)
    {
        cCpleEpip * aCpleE = anArc.attr().mCpleE;
        ELISE_ASSERT(aCpleE!=0,"Could not get Epi in NameImage");
        aRes = aCpleE->LocNameImEpi(aRes);
    }
    return aRes;
}
/*
*/



/*****************************************************************/
/*                                                               */
/*                            cImaMM                             */
/*                                                               */
/*****************************************************************/

cImaMM::cImaMM(const std::string & aName,cAppliWithSetImage & anAppli) :
   mCamGen     (anAppli.CamGenOfName(aName)),
   mCamS       (mCamGen ? mCamGen->DownCastCS() : 0 ),
   mNameIm     (aName),
   mBande      (""),
   mNumInBande (-1),
   mC3         (mCamGen ? mCamGen->OrigineProf() : Pt3dr(0,0,0)),
   mC2         (mC3.x,mC3.y),
   mAppli         (anAppli),
   mPtrTiffStd    (0),
   mPtrTiff8BGr   (0),
   mPtrTiff8BCoul (0),
   mPtrTiff16BGr  (0)
{
}

CamStenope * cImaMM::CamSNN()
{
    ELISE_ASSERT(mCamS!=0,"cImaMM::CamSNN");
    return mCamS;
}
CamStenope * cImaMM::CamSSvp()
{
    return mCamS;
}
cBasicGeomCap3D *  cImaMM::CamGen()
{
   return mCamGen;
}


Tiff_Im  &   cImaMM::TiffStd()
{
    if (mPtrTiffStd==0)
    {
        std::string aFullName =  mAppli.Dir() + mNameIm;
        mPtrTiffStd = new Tiff_Im(Tiff_Im::UnivConvStd(aFullName.c_str()));
    }
    return *mPtrTiffStd;
}


Tiff_Im  &   cImaMM::Tiff16BGr()
{
    if (mPtrTiff16BGr==0)
    {
        std::string aFullName =  mAppli.Dir() + mNameIm;
        mPtrTiff16BGr = new Tiff_Im(Tiff_Im::StdConvGen(aFullName.c_str(),1,true));
    }
    return *mPtrTiff16BGr;
}


/*****************************************************************/
/*                                                               */
/*                      cAppliWithSetImage                       */
/*                                                               */
/*****************************************************************/

/*
class cElemAppliSetFile
{
    public :
       cElemAppliSetFile();
       cElemAppliSetFile(const std::string &);
       void Init(const std::string &);


       std::string mFullName;
       std::string mDir;
       std::string mPat;
       cInterfChantierNameManipulateur * mICNM;
       const cInterfChantierNameManipulateur::tSet * mSetIm;
};
*/

cElemAppliSetFile::cElemAppliSetFile() :
    mICNM(0),
    mSetIm (0)
{
}
cElemAppliSetFile::cElemAppliSetFile(const std::string & aFullName)
{
    Init(aFullName);
}

void cElemAppliSetFile::Init(const std::string & aFullName)
{
    mFullName =aFullName;
#if (ELISE_windows)
        replace( mFullName.begin(), mFullName.end(), '\\', '/' );
#endif
   SplitDirAndFile(mDir,mPat,mFullName);

   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mSetIm = mICNM->Get(mPat);
}

const cInterfChantierNameManipulateur::tSet * cElemAppliSetFile::SetIm()
{
    return mSetIm;
}

/*****************************************************************/
/*                                                               */
/*                      cAppliWithSetImage                       */
/*                                                               */
/*****************************************************************/

std::vector<CamStenope*> cAppliWithSetImage::VCamStenope()
{
    std::vector<CamStenope*> aVC;

    for (int aK=0; aK<int(mVSoms.size()) ; aK++)
    {
        aVC.push_back(mVSoms[aK]->attr().mIma->CamSNN());
    }

    return aVC;
}

std::vector<ElCamera *> cAppliWithSetImage::VCam()
{
    std::vector<ElCamera *> aVC;
    for (int aK=0; aK<int(mVSoms.size()) ; aK++)
    {
        aVC.push_back(mVSoms[aK]->attr().mIma->CamSNN());
    }

    return aVC;
}

const  std::string BLANK(" ");

std::string  cAppliWithSetImage::PatFileOfImSec() const
{
   return ::PatFileOfImSec(mOri);
}

std::string  cAppliWithSetImage::DirAndPatFileOfImSec() const
{
   return ::DirAndPatFileOfImSec(mEASF.mDir,mOri);
}


std::string  cAppliWithSetImage::DirAndPatFileMMByP() const
{
   return ::DirAndPatFileMMByP(mEASF.mDir);
}

void cAppliWithSetImage::Develop(bool EnGray,bool Cons16B)
{
    Paral_Tiff_Dev(mEASF.mDir,*mEASF.SetIm(),(EnGray?1:3),Cons16B);
}


cAppliWithSetImage::cAppliWithSetImage(int argc,char ** argv,int aFlag,const std::string & aNameCAWSI)  :
   mSym       (true),
   mShow      (false),
   mPb        (""),
   mWithCAWSI (aNameCAWSI!=""),
   mAverNbPix (0.0),
   mByEpi     (false),
   mSetMasters(0),
   mCalPerIm  (false),
   mPenPerIm  (-1),
   mModeHelp  (false),
   mNbAlti    (0),
   mSomAlti   (0.0),
   mSetImNoMasq (0)
{
   for (int aK=0 ; aK<argc; aK++)
   {
      if (std::string(argv[aK]) == std::string("-help"))
      {
         mModeHelp = true;
         return;
      }
   }
   if (MMVisualMode) return;

   mWithOri  = ((aFlag & TheFlagNoOri)==0);
   if (argc< (mWithOri ? 2 : 1 ) )
   {
      if (( aFlag & TheFlagAcceptProblem) == 0)
      {
           std::cout << "NbArgs=" << argc << "\n";
           for (int aK=0 ; aK<argc ; aK++)
               std::cout << "ARG[" << aK << "]=" << argv[aK] << "\n";
           ELISE_ASSERT(false,"Not Enough Arg in cAppliWithSetImage");
      }
      mPb = "Not Enough Arg in cAppliWithSetImage";
      return;
   }


   mEASF.Init(argv[0]);

   if (mWithCAWSI)
   {
      cChantierAppliWithSetImage aCAWSI = StdGetFromSI(Dir()+aNameCAWSI,ChantierAppliWithSetImage);
      for (std::list<cCWWSImage>::iterator it=aCAWSI.Images().begin(); it!=aCAWSI.Images().end() ;it++)
          mDicWSI[it->NameIm()] = *it;
   }
/*
   mFullName = argv[0];
#if (ELISE_windows)
        replace( mFullName.begin(), mFullName.end(), '\\', '/' );
#endif
   SplitDirAndFile(mDir,mPat,mFullName);
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mSetIm = mICNM->Get(mPat);
*/


// void cAppliWithSetImage::Develop(bool EnGray,bool Cons16B)
   if (aFlag & TheFlagDev16BGray) Develop(true,true);
   if (aFlag & TheFlagDev8BGray)  Develop(true,false);
   if (aFlag & TheFlagDev8BCoul)  Develop(false,false);
   if (aFlag & TheFlagDevXml)
   {
       MakeXmlXifInfo(mEASF.mFullName,mEASF.mICNM);
   }



   if (mEASF.SetIm()->size()==0)
   {
       std::cout << "For Pat= [" << mEASF.mPat << "]\n";
       ELISE_ASSERT(false,"Empty pattern");
   }

   if (mWithOri)
   {
       mOri = argv[1];
       mEASF.mICNM->CorrecNameOrient(mOri);

   }
   else
   {
       mOri = "NONE";
   }
   mKeyOri =  "NKS-Assoc-Im2Orient@-" + mOri;


   int aNbImGot = 0;
   for (int aKV=0 ; aKV<int(mEASF.SetIm()->size()) ; aKV++)
   {
       const std::string & aName = (*mEASF.SetIm())[aKV];
       if (CAWSI_AcceptIm(aName))
       {
           cImaMM * aNewIma = new cImaMM(aName,*this);
           tSomAWSI & aSom = mGrIm.new_som(cAttrSomAWSI(aNewIma,aKV,aNbImGot));
           mVSoms.push_back(&aSom);
           mDicIm[aName] = & aSom;
/*
       mImages.push_back(new cImaMM(aName,*this));
       mDicIm[aName] = mImages.back();
*/
           Pt2di  aSz =  aNewIma->Tiff16BGr().sz();
           mAverNbPix += double(aSz.x) * double(aSz.y);

           if (mWithOri)
           {
               if (aNewIma->CamGen()->AltisSolIsDef())
               {
                    mSomAlti += aNewIma->CamGen()->GetAltiSol();
                    mNbAlti++;
               }
           }
           aNbImGot++;
       }

   }
   ELISE_ASSERT(aNbImGot!=0,"No image in Appli With Set Image");
   mAverNbPix /= aNbImGot;
}

bool  cAppliWithSetImage::CAWSI_AcceptIm(const std::string & aName) const
{
   return (!mWithCAWSI) || (DicBoolFind(mDicWSI,aName));
}


void cAppliWithSetImage::SaveCAWSI(const std::string & aName)
{
   cListOfName aLON;
   cChantierAppliWithSetImage aCAWSI;
   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
       cCWWSImage aWSI;
       aWSI.NameIm() = (*anITS).attr().mIma->mNameIm;
       for (tItAAWSI itA=(*anITS).begin(mSubGrAll) ; itA.go_on() ; itA++)
       {
           cCWWSIVois  aV;
           aV.NameVois() = (*itA).s2().attr().mIma->mNameIm;
           aWSI.CWWSIVois().push_back(aV);
       }
       aCAWSI.Images().push_back(aWSI);
       aLON.Name().push_back(aWSI.NameIm());
       mVNameFinal.push_back(aWSI.NameIm());
   }
   MakeFileXML(aCAWSI,Dir()+aName);
   MakeFileXML(aLON,Dir()+TheMMByPairNameFiles);


}


std::list<std::pair<std::string,std::string> > cAppliWithSetImage::ExpandCommand(int aNumPat,std::string ArgSup,bool Exe,bool WithDir)
{
    std::list<std::string> aLCom;
    std::list<std::pair<std::string,std::string> >  aRes;
    for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
    {
       std::string aNIm = mVSoms[aK]->attr().mIma->mNameIm;
       std::string aNameDir = WithDir ? (Dir()+aNIm) : aNIm;
       std::string aNCom = SubstArgcArvGlob(aNumPat,aNameDir) + " " + ArgSup;
       aRes.push_back(std::pair<std::string,std::string>(aNCom,aNIm));
       aLCom.push_back(aNCom);
    }
    if (Exe)
       cEl_GPAO::DoComInParal(aLCom);
    return aRes;
}


bool cAppliWithSetImage::HasOri() const
{
   return mWithOri;
}

const std::string & cAppliWithSetImage::Ori() const
{
   return mOri;
}


int cAppliWithSetImage::NbAlti() const
{
   return mNbAlti;
}

double cAppliWithSetImage::AltiMoy() const
{
    ELISE_ASSERT(mNbAlti!=0,"cAppliWithSetImage::AltiMoy => No Alti Init");
    return mSomAlti / mNbAlti;
}

//  aSz * 2 ^ (LogDeZoom * 2) = mAverNbPix;
int  cAppliWithSetImage::DeZoomOfSize(double aSz) const
{
    double aRatio = mAverNbPix / aSz;
    double aRL2 = log2(aRatio) / 2;
    int aL2 = ElMax(0,round_ni(aRL2));
    return 1 << aL2;
}

void cAppliWithSetImage::SuppressSom(tSomAWSI & aSom)
{
    int aNbEr = (int)mDicIm.erase(aSom.attr().mIma->mNameIm);
    ELISE_ASSERT(aNbEr==1,"Incoherence in cAppliWithSetImage::FilterImageIsolated");
    aSom.remove();
}



void cAppliWithSetImage::FilterImageIsolated(bool AnalysConexions)
{
   std::vector<tSomAWSI *> aRes;

   if (mSetImNoMasq)
   {
        for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
        {
              if (! BoolFind(*mSetImNoMasq,(*anITS).attr().mIma->mNameIm))
                 SuppressSom(*anITS);
        }
   }


   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
          if ( AnalysConexions && ((*anITS).nb_succ(mSubGrAll) ==0))
          {
           //std::map<std::string,tSomAWSI *>::iterator itS = mDicIm.find((*anITS).attr().mIma->mNameIm);
           //ELISE_ASSERT(itS!=mDicIm.end(),"Incoherence in cAppliWithSetImage::FilterImageIsolated");
/*
           int aNbEr = mDicIm.erase((*anITS).attr().mIma->mNameIm);
           ELISE_ASSERT(aNbEr==1,"Incoherence in cAppliWithSetImage::FilterImageIsolated");
           (*anITS).remove();
*/
              SuppressSom(*anITS);
          }
          else
          {
              aRes.push_back(&(*anITS));
          }
   }
   mVSoms = aRes;
}



/*
cInterfChantierNameManipulateur
*/
cInterfChantierNameManipulateur * cAppliWithSetImage::ICNM()
{
   return mEASF.mICNM;
}


const std::string & cAppliWithSetImage::Dir() const
{
   return mEASF.mDir;
}
void cAppliWithSetImage::VerifAWSI()
{
   ELISE_ASSERT(mPb=="",mPb.c_str());
}

CamStenope * cAppliWithSetImage::CamOfName(const std::string & aNameIm)
{
   if (mOri=="NONE")
   {
      cOrientationConique anOC = StdGetFromPCP(Basic_XML_MM_File("Template-OrCamAngWithInterne.xml"),OrientationConique);

      // Tiff_Im aTF = Tiff_Im::StdConvGen(mDir+aNameIm,,);
      // Tiff_Im aTF = Tiff_Im::UnivConvStd(mEASF.mDir+aNameIm);
      Tiff_Im aTF = Tiff_Im::StdConvGen(mEASF.mDir+aNameIm,1,true);

      Pt2dr  aSz = Pt2dr(aTF.sz());
      anOC.Interne().Val().F() = euclid(aSz);
      anOC.Interne().Val().PP() = aSz/2.0;
      anOC.Interne().Val().SzIm() = round_ni(aSz);
      anOC.Interne().Val().CalibDistortion()[0].ModRad().Val().CDist() =  aSz/2.0;

      std::string aName = Basic_XML_MM_File("TmpCam"+ GetUnikId() +".xml");
      MakeFileXML(anOC,aName);
      // return Std_Cal_From_File(Basic_XML_MM_File("TmpCam.xml"));
      CamStenope * aRes =  CamOrientGenFromFile(aName,0);
      ELISE_fp::RmFile(aName);

      return aRes;


     // return ;
   }
   std::string aNameOri =  mEASF.mICNM->Assoc1To1(mKeyOri,aNameIm,true);
   return   CamOrientGenFromFile(aNameOri,mEASF.mICNM);
}

cBasicGeomCap3D * cAppliWithSetImage::CamGenOfName(const std::string & aName)
{
    return mWithOri ? mEASF.mICNM->StdCamGenerikOfNames(mOri,aName) : 0;
}

void  cAppliWithSetImage::MakeStripStruct(const std::string & aPairByStrip,bool StripIsFirst)
{

  cElRegex anAutom(aPairByStrip.c_str(),10);
  std::string aExpStrip = StripIsFirst ? "$1" : "$2";
  std::string aExpNumInStrip = StripIsFirst ? "$2" : "$1";

  // for (int aKI=0;  aKI<int(mImages.size()) ; aKI++)
  for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
  {
      // cImaMM & anI = *(mImages[aKI]);
      cImaMM & anI = *((*anITS).attr().mIma);

      std::string aBande = MatchAndReplace(anAutom,anI.mNameIm,aExpStrip);
      std::string aNumInBande = MatchAndReplace(anAutom,anI.mNameIm,aExpNumInStrip);

      bool OkNum = FromString(anI.mNumInBande,aNumInBande);
      ELISE_ASSERT(OkNum,"Num in bande is not numeric");
      if (mShow)
         std::cout << " Image " << anI.mNameIm << " belongs to strip " << aBande <<  " and its number in the strip is " << anI.mNumInBande << "\n";
      anI.mBande = aBande;
  }
}

// Pt2dr PtOfcImaMM   (const cImaMM & aCam) {return aCam.mC2;}
// Pt2dr PtOfcImaMMPtr(const cImaMM * aCam) {return PtOfcImaMM(*aCam);}

void cAppliWithSetImage::AddDelaunayCple()
{
  Delaunay_Mediatrice
  (
      &(mVSoms[0]),
      &(mVSoms[0])+mVSoms.size(),
       PtOfSomAWSIPtr,
       *this,
       1e10,
       (tSomAWSI **) 0
  );

}

void cAppliWithSetImage::AddCoupleMMImSec(bool ExApero,bool SupressImInNoMasq,bool AddCple, const std::string &SetHom, bool ExpTxt,bool ExpImSec,double aTetaOpt)
{
      std::string aCom = MMDir() + "bin/mm3d AperoChImSecMM "
                         + BLANK + QUOTE(mEASF.mFullName)
                         + BLANK + mOri
			 + BLANK + "ExpTxt=" + ToString(ExpTxt)
			 + BLANK + "ExpImSec=" + ToString(ExpImSec)
             + BLANK + "SH=" + SetHom
             + BLANK + "TetaOpt=" + ToString(aTetaOpt);
	 
      if (mPenPerIm>0)
      {
         aCom = aCom + " PenPerIm=" + ToString(mPenPerIm) + " ";
      }
      if (mCalPerIm)
      {
         aCom = aCom + " CalPerIm=true ";
      }
      if (EAMIsInit(&mMasq3D))
      {
           aCom = aCom  + " Masq3D=" + mMasq3D;
      }
      if (ExApero)
      {
         System(aCom,false,true);
      }
      if (SupressImInNoMasq)
      {
           mSetImNoMasq = mEASF.mICNM->Get(PatFileOfImSec());
           // std::string aKS =   "NKS-Set-OfFile@" + mEASF.mDir+ "Ori-"+mOri + "/FileImSel.xml";
           // mSetImNoMasq = mEASF.mICNM->Get(aKS);
      }


      FilterImageIsolated(false); // MPD 10/06/2015 => Sinon avec les points hors masque cela bugue a l'etape suivante
      if (AddCple)
      {

         for (int aKI=0 ; aKI<int(mVSoms.size()) ; aKI++)
         {
             const std::string & aName1 = mVSoms[aKI]->attr().mIma->mNameIm;
             cImSecOfMaster aISOM = StdGetISOM(mEASF.mICNM,aName1,mOri);
             const std::list<std::string > *  aLIm = GetBestImSec(aISOM,-1,-1,10000,true);
             if (aLIm)
             {
                for (std::list<std::string>::const_iterator itN=aLIm->begin(); itN!=aLIm->end() ; itN++)
                {
                    const std::string & aName2 = *itN;
                    if( ImIsKnown(aName1) && ImIsKnown(aName2))
                       AddPair(ImOfName(aName1),ImOfName(aName2));
                }
             }
         }
      }

}

void cAppliWithSetImage::ComputeStripPair(int aDif)
{
    for (tItSAWSI itS1=mGrIm.begin(mSubGrAll); itS1.go_on() ; itS1++)
    {
        cImaMM & anI1 = *((*itS1).attr().mIma);
        for (tItSAWSI itS2=mGrIm.begin(mSubGrAll); itS2.go_on() ; itS2++)
        {
            cImaMM & anI2 = *((*itS2).attr().mIma);
            if (anI1.mBande==anI2.mBande)
            {
               int aN1 = anI1.mNumInBande;
               int aN2 = anI2.mNumInBande;
               if ((aN1>aN2) && (aN1<=aN2+aDif))
               {
                    bool OK = true;
                    if (OK && EAMIsInit(&mTetaBande))
                    {
                       Pt3dr aV3 = anI2.CamGen()->OrigineProf() - anI1.CamGen()->OrigineProf();
                       Pt2dr aV2(aV3.x,aV3.y);
                       aV2 = vunit(aV2);
                       Pt2dr aDirS = Pt2dr::FromPolar(1,mTetaBande * (PI/180));
                       double aTeta = ElAbs(angle_de_droite(aV2,aDirS));
                       OK = (aTeta < (PI/4));
                    }

                    if (OK)
                    {
                        AddPair(&(*itS1),&(*itS2));
                    }
               }
            }
        }
    }
}


void cAppliWithSetImage::operator()(tSomAWSI* anI1,tSomAWSI* anI2,bool)   // Delaunay call back
{
     AddPair(anI1,anI2);
}
bool cAppliWithSetImage::MasterSelected(const std::string & aName) const
{
   return  (mSetMasters==0) || (mSetMasters->IsSetIn(aName));
}
bool cAppliWithSetImage::MasterSelected(tSomAWSI* aSom) const
{
    return MasterSelected(aSom->attr().mIma->mNameIm);
}

bool  cAppliWithSetImage::CpleHasMasterSelected(tSomAWSI* aS1,tSomAWSI* aS2) const
{
    return MasterSelected(aS1) || MasterSelected(aS2);
}


void cAppliWithSetImage::AddFilePair(const std::string & aFilePair)
{
     cSauvegardeNamedRel aSNR = StdGetFromPCP(mEASF.mDir+aFilePair,SauvegardeNamedRel);
     for
     (
         std::vector<cCpleString>::const_iterator itC=aSNR.Cple().begin();
         itC!=aSNR.Cple().end();
         itC++
     )
     {
         AddPair(itC->N1(),itC->N2(),true);
     }

}


void cAppliWithSetImage::AddLinePair(int aDif, bool ExpTxt)
{
    for (tItSAWSI it1=mGrIm.begin(mSubGrAll); it1.go_on() ; it1++)
    {
        //image 1
        cImaMM & anI1 = *((*it1).attr().mIma);
        for (tItSAWSI it2=mGrIm.begin(mSubGrAll); it2.go_on() ; it2++)
        {
            //image 2
            cImaMM & anI2 = *((*it2).attr().mIma);

            std::string aName1(anI1.mNameIm);
            std::string aName2(anI2.mNameIm);
            int aN1 = 0;
            int aN2 = 0;
            //retreive the numeric part of the two image names in order to compare them.
            for (int i = 0; aName1[i]; ++i)
                if (aName1[i] >= '0' && aName1[i] <= '9' )
                aN1 = aN1 * 10 + (aName1[i] - '0');
            for (int i = 0; aName2[i]; ++i)
                if ( aName2[i] >= '0' && aName2[i] <= '9' )
                aN2 = aN2 * 10 + (aName2[i] - '0');
            int ecart = std::abs(aN1-aN2);

            // test if numerical value from the image name are closed each other
            if ((aN1>aN2) && (ecart<=aDif))
            {
		// tie points should exist for this image pair otherwise the process bug later (during create epip step)
		std::string aNameHom =  mEASF.mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+std::string(ExpTxt?"@txt":"@dat"),aName1,aName2,true);
		if (ELISE_fp::exist_file(aNameHom))
		{
                 AddPair(&(*it1),&(*it2));
                 std::cout << "Adding the following image pair: " << aName1 << " and " << aName2 << " \n";
		}
             }

             if ((aN1=0) || (aN2=0))
                ELISE_ASSERT(false,"Cannot extrat numeric value from image names (in order to determine pair of subsequent images)");
         }
     }
     // todo; warning message if no couple found, or if no numeric part in image name
}



void cAppliWithSetImage::AddPair(const std::string & aN1,const std::string & aN2,bool aSVP)
{
    if (ImIsKnown(aN1) && ImIsKnown(aN2))
    {
         AddPair(ImOfName(aN1),ImOfName(aN2));
         return;
    }
    if (aSVP) return;
    std::cout << "For Names " << aN1 << " " << aN2 << " \n";
    ELISE_ASSERT(false,"cannot associate images in cAppliWithSetImage::AddPair")
    // ImIsKnown
}



void cAppliWithSetImage::AddPair(tSomAWSI * aS1,tSomAWSI * aS2)
{
    if (!(CpleHasMasterSelected(aS1,aS2))) return;


    if (mGrIm.arc_s1s2(*aS1,*aS2))
       return;
    if (aS1->attr().mIma->mNameIm>aS2->attr().mIma->mNameIm)
       ElSwap(aS1,aS2);



    cImaMM * anI1 = aS1->attr().mIma;
    cImaMM * anI2 = aS2->attr().mIma;

    cCpleEpip * aCpleE = 0;
    if (mByEpi) // (mByMM1P)
    {

       aCpleE = StdCpleEpip(mEASF.mDir,mOri,anI1->mNameIm,anI2->mNameIm);

       if (! aCpleE->Ok()) return;
       if (aCpleE->RatioCam() <0.1) return;

       Pt2dr aRatio =  aCpleE->RatioExp();
       double aSeuil = 1.8;
       if ((aRatio.x>aSeuil) || (aRatio.y>aSeuil))
           return;

    }


    mGrIm.add_arc(*aS1,*aS2,cAttrArcAWSI(aCpleE));

}

/*
void cAppliWithSetImage::AddPairASym(cImaMM * anI1,cImaMM * anI2,cCpleEpip * aCpleE)
{
    tPairIm aPair(anI1,anI2);

    if (mPairs.find(aPair) != mPairs.end())
       return;

    mPairs.insert(aPair);

    anI1->mVois.push_back(new cAttrVoisImaMM(anI2,aCpleE));

    if (mShow)
       std::cout << "Add Pair " << anI1->mNameIm << " " << anI2->mNameIm << "\n";
}
*/

void cAppliWithSetImage::DoPyram()
{
    std::string aCom =    MMBinFile(MM3DStr) + " MMPyram " + QUOTE(mEASF.mFullName) + " " + mOri;
    if (mShow)
       std::cout << aCom << "\n";
    System(aCom,false,true);
}

bool cAppliWithSetImage::ImIsKnown(const std::string & aName) const
{
     return DicBoolFind(mDicIm,aName);
}


tSomAWSI * cAppliWithSetImage::ImOfName(const std::string & aName)
{
    tSomAWSI * aRes = mDicIm[aName];
    if (aRes==0)
    {
       std::cout << "For name = " << aName  << " DicoSize =" << mDicIm.size()<< "\n";
       ELISE_ASSERT(false,"Cannot get image");
    }
    return aRes;
}
/*****************************************************************/
/*                                                               */
/*              cAppliClipChantier                               */
/*                                                               */
/*****************************************************************/


cAppliClipChantier::cAppliClipChantier(int argc,char ** argv) :
    cAppliWithSetImage (argc-1,argv+1,0)
{
  std::string aPrefClip = "Cliped";
  std::string aOriOut;
  double      aMinSz = 500;


  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistDirOri)
                    << EAMC(mNameMasterIm,"Image corresponding to the box", eSAM_IsPatFile)
                    << EAMC(mBox,"Box to clip"),
        LArgMain()  << EAM(aPrefClip,"PrefCliped",true,"def= Cliped")
                    << EAM(aOriOut,"OriOut",true,"Out Orientation, def = input")
                    << EAM(aMinSz,"MinSz",true,"Min sz to select cliped def = 500")
   );

  if (!MMVisualMode)
  {
      std::string aDirOut = "Dir-" + aPrefClip + "/";
      StdCorrecNameOrient(mOri,DirOfFile(mEASF.mFullName));


      if (!EAMIsInit(&aOriOut))
         aOriOut = mOri + "-" + aPrefClip;

      mMasterIm  =  ImOfName(mNameMasterIm);

      double aZ = mMasterIm->attr().mIma->CamGen()->GetAltiSol();

      Pt2di aCornIm[4];
      mBox.Corners(aCornIm);

      std::vector<Pt3dr>  mVIm;

      for (int aK=0 ; aK < 4 ; aK++)
      {
          mVIm.push_back(mMasterIm->attr().mIma->CamGen()->ImEtZ2Terrain(Pt2dr(aCornIm[aK]),aZ));
      }

      ELISE_fp::MkDirRec(mEASF.mDir+aDirOut);
      bool DoMoveOri = false;
      std::string aDirOriOut;

   // for (int aKIm = 0 ; aKIm <int(mImages.size()) ; aKIm++)
      for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
      {
          cImaMM & anI = *((*anITS).attr().mIma);
          cBasicGeomCap3D * aCG = anI.CamGen();
       // Pt2dr aP1(0,0);
       // Pt2dr aP0 = Pt2dr(aCS->Sz());
          Pt2di aP0(1e9,1e9);
          Pt2di aP1(-1e9,-1e9);

          for (int aKP=0 ; aKP < 4 ; aKP++)
          {
              Pt2di aPIm = round_ni(aCG->Ter2Capteur(mVIm[aKP]));
              aP0.SetInf(aPIm);
              aP1.SetSup(aPIm);
          }
          Box2di aBoxIm(aP0,aP1);
          Box2di aBoxCam(Pt2di(0,0),Pt2di(aCG->SzPixel()));


          if (! InterVide(aBoxIm,aBoxCam))
          {
             Box2di aBoxRes = Inf(aBoxIm,aBoxCam);
             Pt2di aDec = aBoxRes._p0;
             Pt2di aSZ = aBoxRes.sz();
             std::string aNewIm = aPrefClip + "-" + anI.mNameIm;
             aNewIm = StdPrefix(aNewIm) + ".tif";


             if ((aSZ.x>aMinSz) && (aSZ.y>aMinSz))
             {
                  std::string aCom =      MMBinFile(MM3DStr)
                                     + " ClipIm "
                                     + mEASF.mDir + anI.mNameIm + BLANK
                                     + ToString(aDec) + BLANK
                                     + ToString(aSZ) + BLANK
                                     + " Out=" + aNewIm;

                  System(aCom,false,true);
                  // ELISE_fp::MvFile(mEASF.mDir + aNewIm , mEASF.mDir+aDirOut + aNewIm);

                  std::string aNameOriOut = aCG->Save2XmlStdMMName(mEASF.mICNM,aOriOut,aNewIm,Pt2dr(aDec));
                  ELISE_fp::MvFile(mEASF.mDir + aNewIm , mEASF.mDir+aDirOut + aNewIm);
                  aDirOriOut = DirOfFile(aNameOriOut);
                  DoMoveOri = true;

               // std::cout << "NameOriOut= " << aNameOriOut << "\n";
/*
               if (aCS)
               {
                    aCS->cBasicGeomCap3D :: Save2XmlStdMMName(mEASF.mICNM,aOriOut,aNewIm,Pt2dr(aDec));
                    std::cout << "Box " << anI.mNameIm << aDec << aSZ << "\n";

                    cOrientationConique  aCO = aCS->StdExportCalibGlob();

                    std::string aNameOut =  mEASF.mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-" + aOriOut,aNewIm,true);
                    cCalibrationInternConique * aCIO = aCO.Interne().PtrVal();
                    cCalibrationInterneRadiale * aMR =aCIO->CalibDistortion().back().ModRad().PtrVal();
                    if (1)
                    {
                          ElAffin2D aM2C0 = Xml2EL(aCO.OrIntImaM2C());
                          ElAffin2D  aM2CCliped = ElAffin2D::trans(-Pt2dr(aDec))   * aM2C0;
                          aCO.OrIntImaM2C().SetVal(El2Xml(aM2CCliped));
                      // Sinon ca ne marche pas pour le match
                          aCO.Interne().Val().PixelSzIm().SetVal(Pt2dr(aSZ));
                // aCO.Interne().Val().SzIm() = aSZ;
                    }
                    else
                    {
                         if (aMR)
                         {
                             aCIO->PP() = aCIO->PP() - Pt2dr(aDec);
                             aMR->CDist() = aMR->CDist() - Pt2dr(aDec);
                             aCO.Interne().Val().SzIm() = aSZ;
                         }
                         else
                         {
                             cOrIntGlob anOIG;
                             anOIG.Affinite() = El2Xml(ElAffin2D::trans(-Pt2dr(aDec)));
                             anOIG.C2M() = false;
                             aCO.Interne().Val().OrIntGlob().SetVal(anOIG);
                         }
                    }
                    MakeFileXML(aCO,aNameOut);
                }
                else
                {
                   std::cout << "CLIP ORIENT TO DO 4 Bundle Gen \n";
                   // ELISE_ASSERT(false,"Unfinished ClipChantier pour pushbroom");
                }
*/
              }
          }
      }
      if (DoMoveOri)
      {
         ELISE_fp::MvFile(mEASF.mDir + aDirOriOut , mEASF.mDir+aDirOut + aDirOriOut);
      }

  }
}

/*****************************************************************/
/*                                                               */
/*              clip_im                                          */
/*                                                               */
/*****************************************************************/

int ClipIm_main(int argc,char ** argv)
{

    std::string aNameIn;
    std::string aNameOut;
    Pt2di P0(0,0);
    Pt2di Sz(0,0);
    int  XMaxNot0 = 100000000;
    int  XMinNot0 = -100000000;
    int  AmplRandValOut =0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIn,"Name of Image")
                    << EAMC(P0,"P0, origin of clip")
                    << EAMC(Sz,"SZ, size of clip")  ,
        LArgMain()  << EAM(aNameOut,"Out",true,"Name of output file")
                    << EAM(XMaxNot0,"XMaxNot0",true,"Value will be zeroed fo x > this coord (given in unclip file)")
                    << EAM(XMinNot0,"XMinNot0",true,"Value will be zeroed fo x <=  this coord (given in unclip file)")
                    << EAM(AmplRandValOut,"AmplRandVout",true,"Generate random value for out, give amplitude")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    // Tiff_Im tiff = Tiff_Im::BasicConvStd(aNameIn.c_str());
    Tiff_Im tiff = Tiff_Im::UnivConvStd(aNameIn.c_str());


    if (aNameOut == "")
    {
       if (IsPostfixed(aNameIn))
          aNameOut = StdPrefix(aNameIn)+std::string("_Clip.tif");
       else
          aNameOut = aNameIn+std::string("_Clip.tif");
    }

    L_Arg_Opt_Tiff aLArg = Tiff_Im::Empty_ARG;
    aLArg = aLArg + Arg_Tiff(Tiff_Im::ANoStrip());


    Tiff_Im TiffOut  =     (tiff.phot_interp() == Tiff_Im::RGBPalette)  ?
                           Tiff_Im
                           (
                              aNameOut.c_str(),
                              Sz,
                              tiff.type_el(),
                              Tiff_Im::No_Compr,
                              tiff.pal(),
                              aLArg
                          )                    :
                           Tiff_Im
                           (
                              aNameOut.c_str(),
                              Sz,
                              tiff.type_el(),
                              Tiff_Im::No_Compr,
                              tiff.phot_interp(),
                              aLArg
                          );

    
    bool RandOut = EAMIsInit(&AmplRandValOut);
    Fonc_Num aFoncIn = tiff.in(RandOut ? -1 : 0);
    if (EAMIsInit(&XMaxNot0))
    {
       aFoncIn = aFoncIn * (FX<XMaxNot0);
       if (RandOut) 
          aFoncIn = aFoncIn - (FX>=XMaxNot0);  // Add -1 in this out rect
    }
    if (EAMIsInit(&XMinNot0))
    {
       aFoncIn = aFoncIn * (FX>=XMinNot0);
       if (RandOut) 
          aFoncIn = aFoncIn - (FX<XMinNot0);  // Add -1 in this out rect
    }

 
    if (RandOut)
    {
       Symb_FNum aFIn(aFoncIn);
       aFoncIn =  aFIn * (aFIn>=0)  +  (aFIn<0) * frandr() * AmplRandValOut;
    }

    ELISE_COPY
    (
         TiffOut.all_pts(),
         trans(aFoncIn,P0),
         TiffOut.out()
    );

     return 0;
}



/*****************************************************************/
/*                                                               */
/*              cAppliMMByPair                                   */
/*                                                               */
/*****************************************************************/


cAppliMMByPair::cAppliMMByPair(int argc,char ** argv) :
    cAppliWithSetImage (argc-2,argv+2, TheFlagDev16BGray|TheFlagAcceptProblem),
    mDo           ("APMCRF"),
    mZoom0        (64),
    mZoomF        (1),
    mParalMMIndiv (false),
    mDelaunay     (false),
    mRunAperoImSec(false),
    mAddCpleImSec (false),
    mAddCpleLine  (false),
    mDiffInStrip  (1),
    mStripIsFirt  (true),
    mDirBasc      ("MTD-Nuage"),
    mIntIncert    (1.25),
    mSkipCorDone  (false),
    mByMM1P       (true),
    mStrQualOr    ("Low"),
    mHasVeget     (false),
    mSkyBackGround(true),
    mDoOMF        (false),
    mRIEInParal   (true),
    mRIE2Do       (true),
    mExeRIE       (true),
    mDoTiePM0     (false),
    mTimes        (1),
    mDebugCreatE  (false),
    mDebugMMByP    (false),
    mPurge        (! MPD_MM()),
    mUseGpu        (false),
    mDefCor        (0.5),
    mZReg          (0.05),
    mExpTxt        (false),
    mExpImSec      (true),
    mSuprImNoMasq  (false),
    mPIMsDirName   ("Statue"), // used in MMEnvStatute for differenciating PIMs-Forest from PIMs-Statue
    mSetHom        (""),
    mTetaOpt       (0.17)
{
  if ((argc>=2) && (!mModeHelp))
  {
     ELISE_ASSERT(argc >= 2,"Not enough arg");
     mStrType = argv[1];
     StdReadEnum(mModeHelp,mType,mStrType,eNbTypeMMByP);

     mMacType = IsMacType(mType);


     if (mType==eGround)
     {
        mStrQualOr = "High";
        mHasVeget = true;
        mSkyBackGround = false;
        mDelaunay = true;
        mRIE2Do = false;
     }
     else if (mType==eStatue)
     {
        mStrQualOr = "High"; // Depuis les essais de Calib Per Im, il semble pas besoin de ca ?
        mAddCpleImSec = true;
        mHasVeget = false;
        mSkyBackGround = true;
        mRIE2Do = true;
        mZoomF = 4;
        mSuprImNoMasq = true;
     }
     else if (mType==eForest)
     {
        mStrQualOr = "High";
        // do not add the segondary images computed by apero, but do the computation of pair because some data are required anyway (for mask computation based on tie points for e.g)
        // mAddCpleImSec = false;
        // Modif MPD  16/10/2017 car sinon ca plante quand forest est utilise sans FilePair
        // vu lors du stage terrain a Murol (arbre sur muraille), suite demande Antoine Pinte
        mAddCpleImSec = true;
        // do the computation whitout adding the pairs
        mRunAperoImSec= true;
        mHasVeget = true;
        mSkyBackGround = false;
        mRIE2Do = true;
        //mZoom0 = 32;
        mZoomF = 4;
        mSuprImNoMasq = true;
        mDefCor = 0.2;
        mZReg   = 0.02;
        mPIMsDirName="Forest"; // for MMEnvStatute
        mAddCpleLine=true;
     }
     else if (mMacType)
     {
        mStrQualOr = "High"; // Depuis les essais de Calib Per Im, il semble pas besoin de ca ?
        mAddCpleImSec = true;
        mHasVeget = false;
        mSkyBackGround = true;
        mRIE2Do = false;
        mZoom0 = 4;
        mZoomF = 4;
        mDoTiePM0 = true;
        mSuprImNoMasq = true;
     }
     else if (mType==eTestIGN)
     {
        mStrQualOr = "High";
        mDelaunay = true;
        mRIE2Do = true;
     }
  }


  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mStrType,"Type in enumerated values", eSAM_None,ListOfVal(eNbTypeMMByP))
                    << EAMC(mEASF.mFullName,"Full Name (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistDirOri),
        LArgMain()  << EAM(mZoom0,"Zoom0",true,"Zoom Init, Def=64",eSAM_IsPowerOf2)
                    << EAM(mZoomF,"ZoomF",true,"Zoom Final, Def=1",eSAM_IsPowerOf2)
                    << EAM(mDelaunay,"Delaunay",true,"Add Delaunay edges in pair to match, Def=true on ground")
                    << EAM(mFilePair,"FilePair",true,"Add a File containing explicit image pair (as in Tapioca, a <SauvegardeNamedRel> struct ...")
                    << EAM(mAddCpleImSec,"MMImSec",true,"Add pair from AperoChImSecMM,  Def=true in mode Statue")
                    << EAM(mAddCpleLine,"ImLine",true,"Add pair for successive images, based on the numeric value in the image name, Def=true in mode Forest")
                    << EAM(mPairByStrip,"ByStrip",true,"Pair in same strip, first () : strip, second () : num in strip (or reverse with StripIsFisrt)")
                    << EAM(mStripIsFirt,"StripIsFisrt",true,"If true : first expr is strip, second is num in strip Def=true")
                    << EAM(mDiffInStrip,"DeltaStrip",true,"Delta in same strip (Def=1,apply with mPairByStrip)")
                    << EAM(mSym,"Sym",true,"Symetrise all pair (Def=true)")
                    << EAM(mShow,"Show",true,"Show details (def = false))")
                    << EAM(mIntIncert,"Inc",true,"Uncertainty interval for matching")
                    << EAM(mTetaBande,"TetaStrip",true,"If used, cut strip when dir of vector > 45 degree from TetaStrip")
                    << EAM(mSkipCorDone,"SMD",true,"Skip Matching When Already Done (Def=false)")
                    << EAM(mDo,"Do",true,"Step to Do in [Apero-Ch-Im,Pyram,MetaData,Correl,Reech,Fusion,inspect], Def \"APMCFR\" (i.e. All Step)")
                    << EAM(mByMM1P,"ByMM1P",true,"Do match using new MM1P, def = true")
                    << EAM(mImageOfBox,"ImOfBox",true,"Image to define box for MTD (test purpose to limit size of result)")
                    << EAM(mBoxOfImage,"BoxOfIm",true,"Associated to ImOfBox, def = full")
                    << EAM(mParalMMIndiv,"ParMMI",true,"If true each MM if // (\" expert\" option, Def=false currently)")
                    << EAM(mStrQualOr,"QualOr",true,"Quality orient (in High, Average, Low, Def= Low with statue)",eSAM_None,ListOfVal(eNbTypeQual,"eQual_"))
                    << EAM(mRIEInParal,"RIEPar",true,"Internal use (debug Reech Inv Epip)", eSAM_InternalUse)
                    << EAM(mTimes,"TimesExe",true,"Internal use (debug Reech Inv Epip)", eSAM_InternalUse)
                    << EAM(mDebugCreatE,"DCE",true,"Debug Create Epip", eSAM_InternalUse)
                    << EAM(mDebugMMByP,"DebugMMByP",true,"Debug this programm", eSAM_InternalUse)
                    << EAM(mDoOMF,"DoOMF",true,"Do Only Masq Final (tuning purpose)")
                    << EAM(mHasVeget,"HasVeg",true,"Scene contains vegetation (Def=true on Ground)")
                    << EAM(mSkyBackGround,"HasSBG",true,"Scene has sky (or homogeneous) background (Def=false on Ground)")
                    << EAM(mMasterImages,"Masters",true,"If specified, only pair containing a master will be selected")
                    << EAM(mMasq3D,"Masq3D",true,"If specified the 3D masq")
                    << EAM(mCalPerIm,"CalPerIm",true,"true id Calib per Im were used, def=false")
                    << EAM(mPenPerIm,"PenPerIm",true,"Penality Per Image in choice im sec")
                    << EAM(mPurge,"Purge",true,"Purge unused temporay files (Def=true, may be incomplete during some times)")
                    << EAM(mUseGpu,"UseGpu",false,"Use cuda (Def=false)")
                    << EAM(mDefCor,"DefCor",false,"Def corr (context condepend 0.5 Statue, 0.2 Forest)")
                    << EAM(mZReg,"ZReg",true,"Z Regul (context condepend,  0.05 Statue, 0.02 Forest)")
   		            << EAM(mExpTxt,"ExpTxt",false,"Use txt tie points for determining image pairs and/or computing epipolar geometry (Def false, e.g. use dat format)")
   		            << EAM(mSetHom,"SH",false,"Set of Hom, Def=\"\"")
   		            << EAM(mExpImSec,"ExpImSec",false,"Export ImSec def=true (put false if set elsewhere)")
                    << EAM(mTetaOpt,"TetaOpt",true,"For the choice of secondary images: Optimal angle of stereoscopy, in radian, def=0.17 (+or- 10 degree)")
  );

  // Par defaut c'est le meme comportement
    if ((!EAMIsInit(&mRunAperoImSec)) && (!mRunAperoImSec))
        mRunAperoImSec=mAddCpleImSec;
	
  if (!MMVisualMode)
  {
      if (EAMIsInit(&mFilePair))
      {
          if (!EAMIsInit(&mDelaunay)) mDelaunay = false;
          if (!EAMIsInit(&mAddCpleImSec)) mAddCpleImSec = false;
      }

      mExeRIE = mRIE2Do;

      if (EAMIsInit(&mMasterImages))
         mSetMasters =  mEASF.mICNM->KeyOrPatSelector(mMasterImages);
      if (! BoolFind(mDo,'R'))
         mExeRIE = false;

      StdCorrecNameOrient(mOri,DirOfFile(mEASF.mFullName));


      mByEpi = mByMM1P && (! mMacType);

      mQualOr = Str2eTypeQuality("eQual_"+mStrQualOr);


      if (mAddCpleLine)
      {
          AddLinePair(1,mExpTxt);
      }
    

      if (mModeHelp)
          StdEXIT(0);
      if ((! EAMIsInit(&mZoom0))  && (! mMacType))
         mZoom0 =  DeZoomOfSize(7e4);
      VerifAWSI();

      if (EAMIsInit(&mPairByStrip))
      {
          MakeStripStruct(mPairByStrip,mStripIsFirt);
          ComputeStripPair(mDiffInStrip);
      }
      if (mDelaunay)
         AddDelaunayCple();
      if (mRunAperoImSec)
      {
         AddCoupleMMImSec(BoolFind(mDo,'A'),mSuprImNoMasq,mAddCpleImSec,mSetHom,mExpTxt,mExpImSec,mTetaOpt);
      }

      if (EAMIsInit(&mFilePair))
      {
          AddFilePair(mFilePair);
      }
      FilterImageIsolated();

      // mVSoms

      mNbStep = round_ni(log2(mZoom0/double(mZoomF))) + 3 ;

      SaveCAWSI(TheMMByPairNameCAWSI);

      // Paral_Tiff_Dev(mEASF.mDir,mVNameFinal,1,true);
      // Paral_Tiff_Dev(mEASF.mDir,mVNameFinal,3,false); // On en aura sans doute besoin tot ou tard

  }
}


void cAppliMMByPair::DoCorrelEpip()
{
   std::list<std::string> aLCom;
   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
        for (tItAAWSI itA=(*anITS).begin(mSubGrAll) ; itA.go_on() ; itA++)
        {
             // cImaMM & anI2 = *(itP->second);
             bool ToDo,Done,Begun;
             std::string aCom =  MatchEpipOnePair(*itA,ToDo,Done,Begun);
             if (aCom != "")
             {
                if (mDebugMMByP)
                   std::cout << "CommMM1P: " << aCom << "\n";
                else
                   aLCom.push_back(aCom);
             }
        }
   }
   if (mDebugMMByP)
   {
       std::cout << "Debug MMByP : Enter to exit\n";
       getchar();
       exit(EXIT_SUCCESS);
   }
   if (mParalMMIndiv)
   {
        cEl_GPAO::DoComInSerie(aLCom);
   }
   else
   {
        cEl_GPAO::DoComInParal(aLCom);
   }
}

void cAppliMMByPair::DoReechantEpipInv()
{
   std::list<std::string> aLCom;
   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
        for (tItAAWSI itA=(*anITS).begin(mSubGrAll) ; itA.go_on() ; itA++)
        {
             std::string aCom =   MMBinFile(MM3DStr) + " TestLib RIE "
                                + BLANK + (*itA).s1().attr().mIma->mNameIm
                                + BLANK + (*itA).s2().attr().mIma->mNameIm
                                + BLANK + mOri
                                + BLANK + " Dir=" + mEASF.mDir;
              aLCom.push_back(aCom);

        }
   }
   cEl_GPAO::DoComInParal(aLCom);
}



std::string cAppliMMByPair::DirMTDImage(const tSomAWSI & aSom) const
{
    return  mEASF.mDir +  TheDIRMergeEPI()  +  aSom.attr().mIma->mNameIm + "/";
}

bool cAppliMMByPair::InspectMTD(tArcAWSI & anArc,const std::string & aName )
{
    std::string  aNameFile =  DirMTDImage(anArc.s1())
                               + aName + "-" + anArc.s2().attr().mIma->mNameIm + ".tif";

    return  ELISE_fp::exist_file(aNameFile);
}

bool cAppliMMByPair::InspectMTD(tArcAWSI & anArc)
{

   return    InspectMTD(anArc,"Depth")
          && InspectMTD(anArc,"Dist")
          && InspectMTD(anArc,"Mask");
}

bool cAppliMMByPair::InspectMTD_REC(tArcAWSI & anArc)
{

     if ((mType != eStatue) || (!mRIEInParal))
        return true;

     return InspectMTD(anArc) &&  InspectMTD(anArc.arc_rec());
}



void cAppliMMByPair::Inspect()
{
   int aNbTot = 0;
   int aNbFinishMatch = 0;
   int aNbBegun = 0;
   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
        for (tItAAWSI itA=(*anITS).begin(mSubGrAll) ; itA.go_on() ; itA++)
        {
            bool ToDo,Done,Begun;
            MatchEpipOnePair(*itA,ToDo,Done,Begun);
            if (ToDo)
            {
               aNbTot ++;

               if (Begun)
               {
                  bool DoneMTD = InspectMTD_REC(*itA);
                  aNbBegun ++;
                  if ((!Done) || (!DoneMTD))
                  {
                     std::cout << "Pair unfinished " << (*itA).s1().attr().mIma->mNameIm
                               << "  " << (*itA).s2().attr().mIma->mNameIm
                               << " MTD=[" << InspectMTD(*itA) << "::" <<  InspectMTD((*itA).arc_rec()) << "]"
                               << "\n";
                  }

               }
               if (Done)
               {
                  aNbFinishMatch ++;
               }
            }
        }
   }

   std::cout << "Nb Pair Tot " << aNbTot << " Done " << aNbFinishMatch  << " Begun " << aNbBegun << "\n";
   getchar();
}

std::string cAppliMMByPair::MatchEpipOnePair(tArcAWSI & anArc,bool & ToDo,bool & Done,bool & Begun)
{
     ToDo = false;
     cImaMM & anI1 = *(anArc.s1().attr().mIma);
     cImaMM & anI2 = *(anArc.s2().attr().mIma);
     if (anI1.mNameIm >= anI2.mNameIm)
        return "";
     ToDo = true;

     std::string aMatchCom =     MMBinFile(MM3DStr)
                         +  " MM1P"
                         +  BLANK + anI1.mNameIm
                         +  BLANK + anI2.mNameIm
                         +  BLANK + mOri
                         +  " ZoomF=" + ToString(mZoomF)
                         +  " CreateE=" + ToString(mByEpi)
                         +  " InParal=" + ToString(mParalMMIndiv)
                         +  " QualOr=" +  mStrQualOr
                         +  " DCE=" +  ToString(mDebugCreatE)
                         +  " HasVeg=" + ToString(mHasVeget)
                         +  " HasSBG=" + ToString(mSkyBackGround)
                         +  " PurgeAtEnd=" + ToString(mPurge)
                         +  " UseGpu=" + ToString(mUseGpu)
                         +  " DefCor=" + ToString(mDefCor)
                         +  " ZReg=" + ToString(mZReg)
			 +  " ExpTxt=" + ToString(mExpTxt)
                      ;

     if (EAMIsInit(&mMasq3D)) aMatchCom = aMatchCom + " Masq3D=" +mMasq3D + " ";

     if (mType == eGround)
       aMatchCom = aMatchCom + " BascMTD=MTD-Nuage/NuageImProf_LeChantier_Etape_1.xml ";

     if (  mExeRIE && mRIEInParal)
     {
       aMatchCom = aMatchCom + " RIE=true ";
     }


     if (mDoOMF)
        aMatchCom = aMatchCom + " DoOMF=true";

     std::string aNameIm1 = anI1.mNameIm;
     std::string aNameIm2 = anI2.mNameIm;
     if (mByEpi)
     {
        cCpleEpip * aCpleE = StdCpleEpip(mEASF.mDir,mOri,aNameIm1,aNameIm2);
        aNameIm1 = aCpleE->LocNameImEpi(aNameIm1);
        aNameIm2 = aCpleE->LocNameImEpi(aNameIm2);
     }




     std::vector<std::string> aBascCom;
     std::vector<std::string> aVTarget;
     bool AllDoneMatch = true;
     Begun = true;

     for (int aK= 0 ; aK< 2 ; aK++)
     {
         std::string aDirMatch = mEASF.mDir + LocDirMec2Im((aK==0) ? aNameIm1:aNameIm2,(aK==0) ? aNameIm2:aNameIm1);
         std::string aNuageIn =  aDirMatch          + std::string("Score-AR.tif");
         AllDoneMatch  = AllDoneMatch && ELISE_fp::exist_file(aNuageIn);
         std::string aFileInit =  aDirMatch          + std::string("Correl_LeChantier_Num_0.tif");
         Begun  = Begun ||  ELISE_fp::exist_file(aFileInit);



/*
         std::string aNuageGeom =    mDir +  std::string("MTD-Nuage/NuageImProf_LeChantier_Etape_1.xml");
         std::string aNuageTarget =  mDir +  std::string("MTD-Nuage/Basculed-")
                                          + ((aK==0) ? anI1.mNameIm : anI2.mNameIm )
                                          + "-" + ((aK==0) ? anI2.mNameIm :anI1.mNameIm) + ".xml";


         AllDoneMatch = AllDoneMatch && ELISE_fp::exist_file(aNuageIn);
*/
     }
     Done = AllDoneMatch;

     // std::cout << aMatchCom << "\n";
     if ((!AllDoneMatch) || (! mSkipCorDone))
        return aMatchCom;

     return "";
}




void cAppliMMByPair::DoCorrelAndBasculeStd()
{
   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
        for (tItAAWSI itA=(*anITS).begin(mSubGrAll) ; itA.go_on() ; itA++)
        {
             cImaMM & anI1 = *((*itA).s1().attr().mIma);
             cImaMM & anI2 = *((*itA).s2().attr().mIma);
             // cImaMM & anI2 = *(itP->second);

   // ====   Correlation =========================

             bool DoneCor = false;
             if (mSkipCorDone)
             {
                 std::string aFile = mEASF.mDir + "MEC2Im-" + anI1.mNameIm + "-" + anI2.mNameIm + "/MM_Avancement_LeChantier.xml";

                 if (ELISE_fp::exist_file(aFile))
                 {
                       cMM_EtatAvancement anEAv = StdGetFromMM(aFile,MM_EtatAvancement);
                       DoneCor = anEAv.AllDone();
                 }

           // std::cout << "DOCCC " << aFile << " " << Done << "\n";
             }

             if (! DoneCor)
             {
                 std::string aComCor =    MMBinFile("MICMAC")
                                 +  XML_MM_File("MM-Param2Im.xml")
                                 +  std::string(" WorkDir=") + mEASF.mDir          + BLANK
                                 +  std::string(" +Ori=") + mOri + BLANK
                                 +  std::string(" +Im1=")    + anI1.mNameIm  + BLANK
                                 +  std::string(" +Im2=")    + anI2.mNameIm  + BLANK
                                 +  std::string(" +Zoom0=")  + ToString(mZoom0)  + BLANK
                                 +  std::string(" +ZoomF=")  + ToString(mZoomF)  + BLANK
                               ;

                 if (EAMIsInit(&mIntIncert))
                    aComCor = aComCor + " +MulZMax=" +ToString(mIntIncert);

                 if (mShow)
                    std::cout << aComCor << "\n";
                 System(aComCor,false,true);
             }

   // ====   Bascule =========================

             std::string aPreFileBasc =   mEASF.mDir + mDirBasc +  "/Basculed-"+ anI1.mNameIm + "-" + anI2.mNameIm ;
             bool DoneBasc = false;

             if (mSkipCorDone)
             {
                  std::string aFileBasc = aPreFileBasc + ".xml";
                  if (ELISE_fp::exist_file(aFileBasc))
                  {
                      DoneBasc = true;
                  }
             }

             if (! DoneBasc)
             {

                  std::string aComBasc =    MMBinFile(MM3DStr) + " NuageBascule "
                                    + mEASF.mDir+ "MEC2Im-" + anI1.mNameIm + "-" +  anI2.mNameIm + "/NuageImProf_LeChantier_Etape_" +ToString(mNbStep)+".xml "
                                    + mEASF.mDir + mDirBasc + "/NuageImProf_LeChantier_Etape_1.xml "
                                    + aPreFileBasc + " "
                                 ;
                             //  + mDir + mDirBasc +  "/Basculed-"+ anI1.mNameIm + "-" + anI2.mNameIm + " "

                  if (mShow)
                     std::cout  << aComBasc << "\n";
                 System(aComBasc,false,true);
             }

        }
   }
}

/*
void cAppliMMByPair::DoBascule()
{
   for ( tSetPairIm::const_iterator itP= mPairs.begin(); itP!=mPairs.end() ; itP++)
   {
        cImaMM & anI1 = *(itP->first);
        cImaMM & anI2 = *(itP->second);
        std::string aCom =    MMBinFile(MM3DStr) + " NuageBascule "
                             + mDir+ "MEC2Im-" + anI1.mNameIm + "-" +  anI2.mNameIm + "/NuageImProf_LeChantier_Etape_" +ToString(mNbStep)+".xml "
                             + mDir + mDirBasc + "/NuageImProf_LeChantier_Etape_1.xml "
                             + mDir + mDirBasc +  "/Basculed-"+ anI1.mNameIm + "-" + anI2.mNameIm + " "

                            ;
        if (mShow)
           std::cout  << aCom << "\n";
        System(aCom,false,true);
   }
}
*/

void cAppliMMByPair::DoFusionGround()
{
         std::string aCom =    MMBinFile(MM3DStr) + " MergeDepthMap "
                            +   XML_MM_File("Fusion-MMByP-Ground.xml") + BLANK
                            +   "  WorkDirPFM=" + mEASF.mDir + mDirBasc + "/ ";
         if (mShow)
            std::cout  << aCom << "\n";
         System(aCom,false,true);
}

void cAppliMMByPair::DoFusionEpip()
{
   // cMMByImNM * aMMIN = cMMByImNM::ForGlobMerge(Dir(),1.0,"Statue");
   cMMByImNM * aMMIN = cMMByImNM::ForGlobMerge(Dir(),1.0,mStrType);
   // Merge Depth Map
   if (1)
   {
       std::list<std::string> aLCom;
       for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
       {
            std::string aNameIm = (*anITS).attr().mIma->mNameIm;
            std::string aCom =      MMBinFile(MM3DStr) + " MergeDepthMap "
                             +   BLANK +  XML_MM_File("Fusion-MMByP-Statute.xml")
                             + " WorkDirPFM=" + DirMTDImage(*anITS)
                             + " +ImMaster=" + aNameIm
                             + " +Target=" + aMMIN->NameFileXml(eTMIN_Depth,aNameIm)
                           ;

            aLCom.push_back(aCom);
            if (mShow)
                std::cout << aCom << "\n";

       }
       cEl_GPAO::DoComInParal(aLCom);
   }
   else
   {
       for(int aK=0 ; aK<20 ; aK++) std::cout << "SKIPP (tmp) MergeDepthMap , enter to go on\n";
       getchar();
   }


   // Calcul d'une enveloppe qui tienne compte de merge depth map
   {
       std::list<std::string> aLCom;
       for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
       {
            std::string aNameIm = (*anITS).attr().mIma->mNameIm;
            std::string aCom =      MMBinFile(MM3DStr) + " TestLib MMEnvStatute " + aNameIm + " PIMsDirName=" + mPIMsDirName;
            aLCom.push_back(aCom);
            std::cout << aCom << "\n";

       }
       cEl_GPAO::DoComInParal(aLCom);
   }



if (0)
{
   // C'est ce qui concerne la reduction des nuage et images de qualite, pour l'instant pas maintenu ....
    /*
   double aFactRed = 2.0;
   {
       ELISE_fp::MkDir(mEASF.mDir+ DirFusStatue() );
       std::list<std::string> aLComRed;
       for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
       {
            std::string aNameIm = (*anITS).attr().mIma->mNameIm;
            std::string aCom1 =      MMBinFile(MM3DStr) + " ScaleNuage  "
                               + DirMTDImage(*anITS) + "Fusion_"+ aNameIm   + ".xml "
                               + DirFusStatue() + PrefDNF()  +   "Depth" + aNameIm
                               + " " + ToString(aFactRed)
                               + " InDirLoc=false";
                           ;

            std::string aCom2 =  MMBinFile(MM3DStr) + " ScaleIm  "
                               + DirMTDImage(*anITS) + "Fusion_" +  aNameIm + "_Cptr.tif "
                               + " " + ToString(aFactRed)
                               + " Out=" +  DirFusStatue() + PrefDNF() + "Depth" + aNameIm  + "CptRed.tif "
                           ;

             aLComRed.push_back(aCom1);
             aLComRed.push_back(aCom2);

             for (int aK=0 ; aK<2 ; aK++)
             {
                 bool aModeMax = (aK==0);
                 std::string aExt = aModeMax ? "Max" : "Min";
                 std::string aCom = MMBinFile(MM3DStr) + " ScaleNuage  "
                                  + DirMTDImage(*anITS) + "QMNuage-" + aExt + ".xml "
                                  + DirFusStatue() + PrefDNF() +  aExt + aNameIm
                                  +   " " + ToString(aFactRed) + " InDirLoc=false";
                 aLComRed.push_back(aCom);
             }
             // std::cout << aCom2 << "\n";
       }
       cEl_GPAO::DoComInParal(aLComRed);
   }
   */
}






   // getchar();
}

void cAppliMMByPair::DoFusion()
{
    if (mType==eGround)
       DoFusionGround();
    if ((mType==eStatue) || (mType==eForest))
       DoFusionEpip();
}



void cAppliMMByPair::DoMDT()
{
  if (mRIE2Do)
  {
      DoMDTRIE(false);
  }
  if (mDoTiePM0)
  {
     DoMDTRIE(true);
  }
  if (mType==eGround)  DoMDTGround();
}


void cAppliMMByPair::DoMDTRIE(bool ForTieP)
{
   std::list<std::string> aLCOM;
   for (tItSAWSI anITS=mGrIm.begin(mSubGrAll); anITS.go_on() ; anITS++)
   {
            // int aZoom = ForTieP ? mZoom0 : mZoomF;
            int aZoom =  mZoomF;

            cImaMM & anIm = *((*anITS).attr().mIma);
            std::string aCom =     MMBinFile("mm3d MICMAC")
                                +  XML_MM_File("MM-GenMTDFusionImage.xml")
                                +  std::string(" WorkDir=") + mEASF.mDir          + BLANK
                                +  std::string(" +Ori=") + mOri + BLANK
                                +  std::string(" +Zoom=")  + ToString(aZoom)  + BLANK
                                +  " +Im1=" +  anIm.mNameIm + BLANK
                                +  " +PattVois=" +  PatternOfVois(*anITS,true)  + BLANK
                           ;
             if (ForTieP) aCom = aCom + " +PrefixDIR=" + TheDIRMergeEPI();

// std::cout << aCom << "\n";
            aLCOM.push_back(aCom);

             // System(aCom,false,true);
   }
   cEl_GPAO::DoComInParal(aLCOM);
}

void cAppliMMByPair::DoMDTGround()
{
   std::string aCom =     MMBinFile("MICMAC")
                       +  XML_MM_File("MM-GenMTDNuage.xml")
                       +  std::string(" WorkDir=") + mEASF.mDir          + BLANK
                       +  " +PatternAllIm=" +  mEASF.mPat + BLANK
                       +  std::string(" +Ori=") + mOri + BLANK
                       +  std::string(" +Zoom=")  + ToString(mZoomF)  + BLANK
                       +  std::string(" +DirMEC=")  + mDirBasc  + BLANK
                    ;

   if (EAMIsInit(&mImageOfBox))
   {
        Box2di aBox;
        if (EAMIsInit(&mBoxOfImage))
        {
           aBox = mBoxOfImage;
        }
        else
        {
           cImaMM * anIma = ImOfName(mImageOfBox)->attr().mIma;
           aBox = Box2di(Pt2di(0,0),anIma->Tiff16BGr().sz());
        }
        aCom =   aCom
                  + " +WithBox=true"
                  + std::string(" +ImIncluse=") + mImageOfBox
                  + std::string(" +X0=") + ToString(aBox._p0.x)
                  + std::string(" +Y0=") + ToString(aBox._p0.y)
                  + std::string(" +X1=") + ToString(aBox._p1.x)
                  + std::string(" +Y1=") + ToString(aBox._p1.y)
              ;
   }

   System(aCom,false,true);

   std::string aStrN = mEASF.mDir+mDirBasc+"/NuageImProf_LeChantier_Etape_1.xml";
   cXML_ParamNuage3DMaille aNuage = StdGetFromSI(aStrN,XML_ParamNuage3DMaille);
   aNuage.PN3M_Nuage().Image_Profondeur().Val().OrigineAlti() = 0;
   aNuage.PN3M_Nuage().Image_Profondeur().Val().ResolutionAlti() = 1;
   MakeFileXML(aNuage,aStrN);



   std::string aStrZ = mEASF.mDir+mDirBasc+"/Z_Num1_DeZoom"+ToString(mZoomF)+ "_LeChantier.xml";
   cFileOriMnt aFileZ = StdGetFromPCP(aStrZ,FileOriMnt);
   aFileZ.OrigineAlti() = 0;
   aFileZ.ResolutionAlti() = 1;
   MakeFileXML(aFileZ,aStrZ);
}



int cAppliMMByPair::Exe()
{
   for (int aT=0 ; aT<mTimes ; aT++)
   {
       std::string aName = mEASF.mDir+ "cAppliMMByPair_"+ToString(aT);
       FILE * aFP = FopenNN(aName,"w","Indicateur cAppliMMByPair::Exe");
       fclose (aFP);

       if (BoolFind(mDo,'i'))
       {
          Inspect();
       }


       // eQual_Low =>  Pts Hom par match, eQual_Aver => Pts Hom Std
       if (BoolFind(mDo,'P') && ((!mByMM1P) || (mQualOr= eQual_Low)))
       {
          DoPyram();
       }

       if (BoolFind(mDo,'M') )
       {
          DoMDT();
       }
       if (BoolFind(mDo,'C'))
       {
          if (mByMM1P)
          {
             DoCorrelEpip();
          }
          else
          {
             DoCorrelAndBasculeStd();
          }
       }

       if ( BoolFind(mDo,'R') &&  (!mDebugCreatE) &&  (!mRIEInParal) && mExeRIE)
       {
             DoReechantEpipInv();
       }


       if (BoolFind(mDo,'F'))
       {
            DoFusion();
       }

       if (mDebugCreatE)
       {
          ELISE_fp::RmFile(mEASF.mDir+"LockEpi-*.txt");
          ELISE_fp::RmFile(mEASF.mDir+"Epi_Im*");
          ELISE_fp::PurgeDirRecursif(mEASF.mDir+"Homol-DenseM/");
       }
   }
/*
   if (BoolFind(mDo,'F'))
      DoFusion();
*/
   return 0;
}


int MMByPair_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);
   cAppliMMByPair anAppli(argc,argv);

   int aRes = anAppli.Exe();
   BanniereMM3D();
   return aRes;
}


int ChantierClip_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);
   cAppliClipChantier anAppli(argc,argv);

   if (!MMVisualMode) BanniereMM3D();

   return EXIT_SUCCESS;
}
#if (0)
#endif


/************************************************************************/
/*                                                                      */
/*                   Do All Dev                                         */
/*                                                                      */
/************************************************************************/

// int DoAllDev_main(int argc,char ** argv);

/*
class cAppliDoAllDev : cAppliWithSetImage:
cAppliWithSetImage::cAppliWithSetImage(int argc,char ** argv,int aFlag,const std::string & aNameCAWSI)  :
{
     public :

           cAppliDoAllDev(
};
*/

int DoAllDev_main(int argc,char ** argv)
{
    bool  DoDev8BGr  = true;
    bool  DoDev16BGr = true;
    bool  DoDev8BCoul = true;
    bool  DoDevXml   = true;
    std::string      aPat;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aPat,"Pattern of Images", eSAM_IsPatFile),
        LArgMain()  << EAM(DoDev8BGr,"8BGR",true,"Generate 8-bits gray images, def=true")
                    << EAM(DoDev16BGr,"16BGr",true,"Generate 16-bits gray images, def=true")
                    << EAM(DoDev8BCoul,"8BCoul",true,"Generate 8-bits coul images, def=true")
                    << EAM(DoDevXml,"XmlXiff",true,"Generate Xml Xif file, def=true")
    );

    if (MMVisualMode) return EXIT_SUCCESS;

    int AFlag = cAppliWithSetImage::TheFlagNoOri;
    if (DoDev8BGr ) AFlag |= cAppliWithSetImage::TheFlagDev8BGray ;
    if (DoDev16BGr) AFlag |= cAppliWithSetImage::TheFlagDev16BGray;
    if (DoDev8BCoul) AFlag |= cAppliWithSetImage::TheFlagDev8BCoul;
    if (DoDevXml  ) AFlag |= cAppliWithSetImage::TheFlagDevXml;

    cAppliWithSetImage anAppli(argc-1,argv+1,AFlag);

    DoNothingButRemoveWarningUnused(anAppli);

    return EXIT_SUCCESS;
}


void DoAllDev(const std::string & aPat)
{
     std::string aCom =    MMBinFile(MM3DStr) + " AllDev " + QUOTE(aPat);
     System(aCom,false,true);
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
