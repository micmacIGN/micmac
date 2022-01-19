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



#include "NewOri.h"


#define StdNbMaxInit   1500
#define QuickNbMaxInit 600

#define StdDefNbMaxSel   500
#define QuickDefNbMaxSel 200

#define StdTKNbMaxSel   100
#define QuicTKNbMaxSel  30


#define StdNbIterBundle 10
#define QuickNbIterBundle 6

#define  StdNbRansac 200
#define  QuickNbRansac 100


#define ModeShowNbMaxSel   100
#define MaxSurPond3    10.0

#define NbMinPts3TomKan 10


class cAppliOptimTriplet;
class cPairOfTriplet;
class cImOfTriplet ;


class cImOfTriplet
{
   public :
         cImOfTriplet(int aNum,cAppliOptimTriplet &,cNewO_OneIm *,const ElRotation3D & aR);
         cNewO_OneIm * Im() {return mIm;}
         const ElRotation3D & Ori() const {return mRot;}
         ElRotation3D & Ori() {return mRot;}
         void SetOri(const ElRotation3D & aRot) { mRot = aRot;}
         cAppliOptimTriplet &Appli() {return mAppli;}
         std::vector<Pt2df> &  VFullPtOf3() {return mVFullPtOf3;}  // Points triples
         std::vector<Pt2df> &  VRedPtOf3() {return mVRedPtOf3;}  // Points triples
         std::vector<Pt2df> &  VTK_PtOf3() {return mVTK_PtOf3;}  // Points triples
         double Foc() const {return mIm->CS()->Focale();}

#if (ELISE_X11)
         void InitW(const Pt2dr & aSzMax);
#endif
         Video_Win * W() {return mW;}

         void SetReduce(const cResIPR & aResIPR,const cResIPR & aTKIpr);
         int NbR() {return (int)mVRedPtOf3.size();}
         int NbF() {return (int)mVFullPtOf3.size();}
         int  Num() {return mNum;}
   private :
         cImOfTriplet(const cImOfTriplet&); // N.I.
         cAppliOptimTriplet & mAppli;
         cNewO_OneIm * mIm;
         ElRotation3D  mRot;
         std::vector<Pt2df>    mVFullPtOf3;
         std::vector<Pt2df>    mVRedPtOf3;
         std::vector<Pt2df>    mVTK_PtOf3;

         Video_Win *      mW;
         double           mZoomW;
         int              mNum;
};


class cPairOfTriplet
{
     public :
          cPairOfTriplet(cImOfTriplet * aI1,cImOfTriplet *aI2,cImOfTriplet *aI3);
          double ResiduMoy();
          double ResiduMoy(const ElRotation3D & aR1,const ElRotation3D & aR2);

          void AddResidu(const ElRotation3D & aR1,const ElRotation3D & aR2,double &aSomPds,double & aSomPdsRes);
          int NbR() const {return (int)mVRedP1.size();}
          int NbF() const {return (int)mFullVP1.size();}


          const ElRotation3D & Ori1()  const  {return mIm1->Ori();}
          const ElRotation3D & Ori2()  const  {return mIm2->Ori();}
          const tMultiplePF & FullHoms() const  {return mFullHoms;}
          const tMultiplePF & RedHoms() const  {return mRedHoms;}
          const ElRotation3D &  R12Pair () {return mR12Pair;}

          const std::vector<Pt2df>  &   FullVP1 () const {return mFullVP1;};
          const std::vector<Pt2df>  &   FullVP2 () const {return mFullVP2;};
          const std::vector<Pt2df>  &   VRedP1 ()  const {return mVRedP1;};
          const std::vector<Pt2df>  &   VRedP2 ()  const {return mVRedP2;};


          void SetPackRed(bool IsFirst,int aNbMaxInit,int aNbFin,const cResIPR &, const std::vector<Pt2df> & aVRed);

          cImOfTriplet & Im1() {return *mIm1;}
          cImOfTriplet & Im2() {return *mIm2;}
          cImOfTriplet & Im3() {return *mIm3;}

     private :
          cPairOfTriplet(const  cPairOfTriplet&); // N.I.
          cAppliOptimTriplet & mAppli;
          cImOfTriplet *        mIm1;
          cImOfTriplet *        mIm2;
          cImOfTriplet *        mIm3;
          std::vector<Pt2df>    mFullVP1;
          std::vector<Pt2df>    mFullVP2;
          tMultiplePF           mFullHoms;
          tMultiplePF           mRedHoms;

          std::vector<Pt2df>    mVRedP1;
          std::vector<Pt2df>    mVRedP2;
          ElRotation3D          mR12Pair;
};


class cAppliOptimTriplet : public cCommonMartiniAppli
{
      public :
          cAppliOptimTriplet(int argc,char ** argv,bool QuitExit);
          cNewO_NameManager * NM() {return mNM;}
          double ResiduTriplet(const ElRotation3D &,const ElRotation3D &,const ElRotation3D &);
          double ResiduGlob(const ElRotation3D &,const ElRotation3D &,const ElRotation3D &);
          double ResiduTriplet();
          double ResiduGlob();
          bool Show();
          void TestOPA(cPairOfTriplet &);
          void TestSol(const  std::vector<ElRotation3D> & aVR);
 
          static const std::string KeyCple;

      private :
          void ShowPoint(cImOfTriplet * aIm,Pt2df aP,int aCoul,int aRay);
          void ShowPoints(cImOfTriplet * aIm,const std::vector<Pt2df> &,int aCoul,int aRay);
          void ShowPointSel(cImOfTriplet * aIm,Pt2df aP,int aCoul);
          void ShowPointSel(const std::vector<int> &,cImOfTriplet * aIm,const std::vector<Pt2df> &,int aCoul);
          void ShowPointSel(cImOfTriplet * aIm,const std::vector<Pt2df> &,int aCoul);
          void Execute();
          void TestTomasiKanade();

          cAppliOptimTriplet(const cAppliOptimTriplet&); // N.I.
          // std::string mNameOriCalib;
          std::string mDir;
          cNewO_NameManager * mNM;
          std::vector<cImOfTriplet *> mIms;
          std::vector<cPairOfTriplet *> mPairs;
          cImOfTriplet *   mIm1;
          cImOfTriplet *   mIm2;
          cImOfTriplet *   mIm3;
          cPairOfTriplet * mP12;
          cPairOfTriplet * mP13;
          cPairOfTriplet * mP23;

          tMultiplePF      mFullH123;
          tMultiplePF      mRedH123;
          tMultiplePF      mTK_H123;
          double           mFoc;
          cResIPR          mSel3;
          cResIPR          mTKSel3;  // Hyper resuit pour Tomasi Kanade init
          Pt2dr            mSzShow;
          int              mNbMaxSel;
          int              mTKNbMaxSel;
          int              mNbMaxInit;
          bool             mAotShow;
          bool             mCalledBySerie;
          double           mPds3;
          double           mBestResidu;
          std::string      mN1,mN2,mN3;
          bool             mQuitExist;
          cTplTriplet<std::string> m3S;
          cNewO_OneIm *    mNoIm1;
          cNewO_OneIm *    mNoIm2;
          bool             mBugTK;

          bool             mModeTTK;
          bool             mWithTTK;
          bool             mInOriOk;
};

const std::string cAppliOptimTriplet::KeyCple = "KeyCple";


void ReducePts(std::vector<Pt2df> &aVRed,const std::vector<Pt2df>& aVFull,const std::vector<int> & aVInd)
{
    for (int aK=0 ; aK<int(aVInd.size()) ; aK++)
       aVRed.push_back(aVFull[aVInd[aK]]);
}

/**************************************************/
/*                                                */
/*            cImOfTriplet                        */
/*                                                */
/**************************************************/

cImOfTriplet::cImOfTriplet(int aNum,cAppliOptimTriplet & anAppli,cNewO_OneIm * anIm,const ElRotation3D & aR) :
    mAppli (anAppli),
    mIm    (anIm),
    mRot   (aR),
    mNum   (aNum)
{
}
/*
*/
#if (ELISE_X11) 
void cImOfTriplet::InitW(const Pt2dr & aSzMax)
{
      mW  = new Video_Win (Video_Win::LoadTiffWSzMax(mIm->Name(),aSzMax,mZoomW));
      mW->set_title(mIm->Name().c_str());
}
#endif

void  cImOfTriplet::SetReduce(const cResIPR & aResIPR,const cResIPR & aTKIPR)
{
    ReducePts(mVRedPtOf3,mVFullPtOf3,aResIPR.mVSel);
    ReducePts(mVTK_PtOf3,mVFullPtOf3,aTKIPR.mVSel);
}


/**************************************************/
/*                                                */
/*            cPairOfTriplet                      */
/*                                                */
/**************************************************/

cPairOfTriplet::cPairOfTriplet(cImOfTriplet * aI1,cImOfTriplet *aI2,cImOfTriplet *aI3) :
    mAppli    (aI1->Appli()),
    mIm1      (aI1),
    mIm2      (aI2),
    mIm3      (aI3),
    mR12Pair  (ElRotation3D::Id)
{
   mAppli.NM()->LoadHomFloats(mIm1->Im(),mIm2->Im(),&mFullVP1,&mFullVP2);


   mFullHoms.push_back(&mFullVP1);
   mFullHoms.push_back(&mFullVP2);

   if (mAppli.Show())
      std::cout << "cPairOfTriplet " << mFullVP1.size() << " " << mFullVP2.size() << "\n";

   std::string aNameOri = mAppli.NM()->NameXmlOri2Im(mIm1->Im()->Name(),mIm2->Im()->Name(),true);
   cXml_Ori2Im aXmlO =  mAppli.NM()->GetOri2Im(mIm1->Im()->Name(),mIm2->Im()->Name());
   const cXml_O2IRotation & aXO = aXmlO.Geom().Val().OrientAff();
   mR12Pair =    ElRotation3D (aXO.Centre(),ImportMat(aXO.Ori()),true);

}


double cPairOfTriplet::ResiduMoy(const ElRotation3D & aR1,const ElRotation3D & aR2)
{
    std::vector<double> aVRes;
    for (int aK=0 ; aK<int(mFullVP1.size()) ; aK++)
    {
        std::vector<Pt3dr> aW1;
        std::vector<Pt3dr> aW2;
        AddSegOfRot(aW1,aW2,aR1,mFullVP1[aK]);
        AddSegOfRot(aW1,aW2,aR2,mFullVP2[aK]);
        bool OkI;
        Pt3dr aI = InterSeg(aW1,aW2,OkI);
        if (OkI)
        {
            double aRes1 = Residu(mIm1->Im(),aR1,aI,mFullVP1[aK]);
            double aRes2 = Residu(mIm2->Im(),aR2,aI,mFullVP2[aK]);

			if(! std::isnan((aRes1+aRes2)/2.0))
            	aVRes.push_back((aRes1+aRes2)/2.0);

        }
    }
    return MedianeSup(aVRes);
}

void cPairOfTriplet::AddResidu(const ElRotation3D & aR1,const ElRotation3D & aR2,double &aSomPds,double & aSomPdsRes)
{
    double aPds = mFullVP1.size();
    aSomPds += aPds;
    aSomPdsRes += ResiduMoy(aR1,aR2) * aPds;
}

double cPairOfTriplet::ResiduMoy()
{
    return ResiduMoy(Ori1(),Ori2());
}

void cPairOfTriplet::SetPackRed(bool IsFirst,int aNbMaxInit,int aNbFin,const cResIPR & aIPRInit, const std::vector<Pt2df> & aVRed)
{ 
    cResIPR aIPR = IndPackReduit((IsFirst?mFullVP1:mFullVP2),aNbMaxInit,aNbFin,aIPRInit,aVRed);
    ReducePts(mVRedP1,mFullVP1,aIPR.mVSel);
    ReducePts(mVRedP2,mFullVP2,aIPR.mVSel);
    mRedHoms.push_back(&mVRedP1);
    mRedHoms.push_back(&mVRedP2);
}

/**************************************************/
/*                                                */
/*            cAppliOptimTriplet                  */
/*                                                */
/**************************************************/

     //======================= VISU ====================

void cAppliOptimTriplet::ShowPoint(cImOfTriplet * aIm,Pt2df aP,int aCoul,int aRay)
{
     Video_Win * aW = aIm->W();
     if (aW)
     {
        Pt2dr aPr(aP.x,aP.y);
        aPr = aIm->Im()->CS()->R3toF2(PZ1(aPr));
        aW->draw_circle_abs(aPr,aRay,Line_St(aW->pdisc()(aCoul)));
     }
}

void cAppliOptimTriplet::ShowPoints(cImOfTriplet * aIm,const std::vector<Pt2df> & aVPts,int aCoul,int aRay)
{
   for (int aK=0 ; aK<int(aVPts.size()) ; aK++)
       ShowPoint(aIm,aVPts[aK],aCoul,aRay);
}

void cAppliOptimTriplet::ShowPointSel(cImOfTriplet * aIm,Pt2df aP,int aCoul)
{
    ShowPoint(aIm,aP,aCoul,4);
    ShowPoint(aIm,aP,aCoul,6);
    ShowPoint(aIm,aP,aCoul,8);
}
void cAppliOptimTriplet::ShowPointSel(const std::vector<int> & aVSel,cImOfTriplet * aIm,const std::vector<Pt2df> & aVPts,int aCoul)
{
   for (int aK=0 ; aK<int(aVSel.size()) ; aK++)
       ShowPointSel(aIm,aVPts[aVSel[aK]],aCoul);
}

void cAppliOptimTriplet::ShowPointSel(cImOfTriplet * aIm,const std::vector<Pt2df> & aVPts,int aCoul)
{
   for (int aK=0 ; aK<int(aVPts.size()) ; aK++)
       ShowPointSel(aIm,aVPts[aK],aCoul);
}

     //===========================================

void cAppliOptimTriplet::TestSol(const  std::vector<ElRotation3D> & aVR)
{
    // Test un peu heuristique suite au pb de div par zero de la base
    if (BadValue(aVR[0]) || BadValue(aVR[1]) ||  BadValue(aVR[2]))
       return;

    double aResidu = ResiduGlob(aVR[0],aVR[1],aVR[2]);

    // std::cout << "cAppliOptimTriplet::TestSol " << aResidu << "\n";

    if (aResidu < mBestResidu)
    {
       mBestResidu = aResidu;
       for (int aK=0 ; aK<3 ; aK++)
           mIms[aK]->SetOri(aVR[aK]);
    }
}

void cAppliOptimTriplet::TestOPA(cPairOfTriplet & aPair)
{
    ElRotation3D aR1 = ElRotation3D::Id;
    std::list<Appar23> aL32;
    int aI1 = aPair.Im1().Num();
    int aI2 = aPair.Im2().Num();
    int aI3 = aPair.Im3().Num();
    const std::vector<Pt2df> & aVP1 = *(mRedH123[aI1]);
    const std::vector<Pt2df> & aVP2 = *(mRedH123[aI2]);
    const std::vector<Pt2df> & aVP3 = *(mRedH123[aI3]);
    ElRotation3D aR2 = aPair.R12Pair();

    if ( aVP1.size() < 4) return;


    for (int aKP=0 ; aKP<int(aVP1.size()) ; aKP++)
    {
        std::vector<Pt3dr> aVA;
        std::vector<Pt3dr> aVB;

        const Pt2df & aP1 = aVP1[aKP];
        aVA.push_back(Pt3dr(0,0,0));
        aVB.push_back(Pt3dr(aP1.x,aP1.y,1.0));

        const Pt2df & aP2 = aVP2[aKP];
        aVA.push_back(aR2.ImAff(Pt3dr(0,0,0)));
        aVB.push_back(aR2.ImAff(Pt3dr(aP2.x,aP2.y,1.0)));

        bool OkI;
        Pt3dr anInter = InterSeg(aVA,aVB,OkI);
        if (OkI)
        {
            const Pt2df & aP3 = aVP3[aKP];
            aL32.push_back(Appar23(Pt2dr(aP3.x,aP3.y),anInter));
        }
    }

    ElTimer aChrono;
    CamStenopeIdeale aCSI = CamStenopeIdeale::CameraId(true,ElRotation3D::Id);

    double anEcart;
    int aNbRansac =  mQuick ? QuickNbRansac : StdNbRansac;
    ElRotation3D aR3 = aCSI.RansacOFPA(true,aNbRansac,aL32,&anEcart);
    aR3 = aR3.inv();

    std::vector<ElRotation3D> aVR(3,ElRotation3D::Id);
    aVR[aI1] = aR1;
    aVR[aI2] = aR2;
    aVR[aI3] = aR3;

    ElRotation3D aR0 = aVR[0];

    for (int aK=0 ; aK<3 ; aK++)
    {
       aVR[aK] = aR0.inv() * aVR[aK] ;
       // aVR[aK] =  aVR[aK] * aR0.inv() ;
    }
    double aFact = euclid(aVR[1].tr());

    if (aFact < 1e-60) return;

    aVR[1] = ElRotation3D(aVR[1].tr()/aFact,aVR[1].Mat(),true);
    aVR[2] = ElRotation3D(aVR[2].tr()/aFact,aVR[2].Mat(),true);


    TestSol(aVR);
/*
    double aResidu = ResiduGlob(aVR[0],aVR[1],aVR[2]);

    if (aResidu < mBestResidu)
    {
         mBestResidu = aResidu;
         for (int aK=0 ; aK<3 ; aK++)
             mIms[aK]->SetOri(aVR[aK]);
    }
*/
}





bool cAppliOptimTriplet::Show()
{
   return mAotShow;
}

void ShowRot(const std::string & aMes,const ElRotation3D & aR)
{
   std::cout << aMes
             << " O=" << aR.ImAff(Pt3dr(0,0,0))
             << " X=" << aR.ImVect(Pt3dr(1,0,0))
             << " Y=" << aR.ImVect(Pt3dr(0,1,0))
             << " Z=" << aR.ImVect(Pt3dr(0,0,1))
             << "\n";
}

double cAppliOptimTriplet::ResiduTriplet(const ElRotation3D & aR1,const ElRotation3D & aR2,const ElRotation3D & aR3)
{
    
    std::vector<double> aVRes;
    for (int aK=0 ; aK<int(mIm1->VFullPtOf3().size()) ; aK++)
    {
        std::vector<Pt3dr> aW1;
        std::vector<Pt3dr> aW2;
        AddSegOfRot(aW1,aW2,aR1,mIm1->VFullPtOf3()[aK]);
        AddSegOfRot(aW1,aW2,aR2,mIm2->VFullPtOf3()[aK]);
        AddSegOfRot(aW1,aW2,aR3,mIm3->VFullPtOf3()[aK]);
        bool OkI;
        Pt3dr aI = InterSeg(aW1,aW2,OkI);

        if (OkI)
        {
            double aRes1 = Residu(mIm1->Im(),aR1,aI,mIm1->VFullPtOf3()[aK]);
            double aRes2 = Residu(mIm2->Im(),aR2,aI,mIm2->VFullPtOf3()[aK]);
            double aRes3 = Residu(mIm3->Im(),aR3,aI,mIm3->VFullPtOf3()[aK]);

            aVRes.push_back((aRes1+aRes2+aRes3)/3.0);
        }
    }
    return MedianeSup(aVRes);
}

double cAppliOptimTriplet::ResiduTriplet()
{
    return ResiduTriplet(mIm1->Ori(),mIm2->Ori(),mIm3->Ori());
}

double cAppliOptimTriplet::ResiduGlob(const ElRotation3D & aR1,const ElRotation3D & aR2,const ElRotation3D & aR3)
{
    double aSomPds    = 0.0;
    double aSomPdsRes = 0.0;

    double aPds3 = ElMin(MaxSurPond3,(mP12->NbF()+mP13->NbF()+mP23->NbF())/double(mIm1->NbF()));
    aPds3  *=  mIm1->VFullPtOf3().size();

    aSomPds += aPds3;
    aSomPdsRes += ResiduTriplet(aR1,aR2,aR3) * aPds3;

    mP12->AddResidu(aR1,aR2,aSomPds,aSomPdsRes);
    mP13->AddResidu(aR1,aR3,aSomPds,aSomPdsRes);
    mP23->AddResidu(aR2,aR3,aSomPds,aSomPdsRes);

    return aSomPdsRes / aSomPds;
}

double cAppliOptimTriplet::ResiduGlob()
{
    return ResiduGlob(mIm1->Ori(),mIm2->Ori(),mIm3->Ori());
}


cAppliOptimTriplet::cAppliOptimTriplet(int argc,char ** argv,bool QuitExist)  :
    mDir      ("./"),
    // mNbMaxSel (StdDefNbMaxSel),
    mAotShow  (false),
    mCalledBySerie  (false),
    mQuitExist (QuitExist),
    m3S        ("a","b","c"),
    mNoIm1     (0),
    mNoIm2     (0),
    mBugTK     (false),
    mModeTTK    (false),
    mWithTTK    (true)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mN1,"Image one", eSAM_IsExistFile)
                   << EAMC(mN2,"Image two", eSAM_IsExistFile)
                   << EAMC(mN3,"Image three", eSAM_IsExistFile),
        LArgMain() 
                   << EAM(mDir,"Dir",true,"Directory, Def=./ ",eSAM_IsDir)
                   << EAM(mSzShow,"SzShow",true,"Sz of window to show the result in window (Def=none)")
                   << EAM(mNbMaxSel,"NbPts",true,"Nb of selected points")
                   << EAM(mAotShow,"Show",true,"Show Message")
                   << EAM(mCalledBySerie,"CalledBySerie",true,"Called by non paral")
                   << EAM(mBugTK,"BugTK",true,"Debug Tomasi Kanade")
                   << ArgCMA()
   );

   mModeTTK = (ModeNO() ==eModeNO_TTK);
   mWithTTK = (ModeNO() !=eModeNO_StdNoTTK);


   if (mModeTTK)
   {
      mQuitExist = false;
   }

/*
   if (mCalledBySerie)
   {
      mQuitExist = false;
   }
*/

   StdCorrecNameOrient(mNameOriCalib,mDir);


   std::string aNameMin = (mN1<mN2) ? mN1 : mN2;
   std::string aNameMax = (mN1>mN2) ? mN1 : mN2;



   mNM = new cNewO_NameManager(mExtName,mPrefHom,mQuick,mDir,mNameOriCalib,"dat");
   mNoIm1 = new cNewO_OneIm(*mNM,aNameMin);
   mNoIm2 = new cNewO_OneIm(*mNM,aNameMax);

   if (mN3== KeyCple )
   {
      std::string  aNameLON = mNM->NameTripletsOfCple(mNoIm1,mNoIm2,true);
      cListOfName aLON  = StdGetFromPCP(aNameLON,ListOfName);


      for (std::list<std::string>::const_iterator itN =aLON.Name().begin() ; itN!=aLON.Name().end() ; itN++)
      {
         if (mAotShow || mCalledBySerie) 
         {
             std::cout << "============= RUN with  Im3= " << *itN << " ======================\n";
         }
         mN3 = *itN;

         m3S = cTplTriplet<std::string> (mN1,mN2,mN3);
         Execute();
      }
   }
   else
   {
      m3S = cTplTriplet<std::string> (mN1,mN2,mN3);
      mQuitExist = false;
      Execute();
   }
}

void cAppliOptimTriplet::TestTomasiKanade()
{
   if  ((mTK_H123[0]->size() < NbMinPts3TomKan) || (! mWithTTK))
      return;
    double aPrec ;
    if (mModeTTK) 
       mBestResidu= 1e20 ;



    std::vector<ElRotation3D> aVR = OrientTomasiKanade 
                                    (
                                        aPrec,
                                        mTK_H123,
                                        3,
                                        50,
                                        1e-5,
                                        (std::vector<ElRotation3D> *)NULL
                                    );

    // std::cout << "TomasiKanad " << aPrec  <<  "\n";
    TestSol(aVR);
}

void cAppliOptimTriplet::Execute()
{
   if (0 && MPD_MM() && (mN1=="DSC01582.JPG") && (mN2=="DSC01583.JPG") && (mN3=="DSC01606.JPG"))
   {
      std::cout << "Generate bug for testing \n";
      ELISE_ASSERT(false,"Generate bug for testing triplet");
   }


   ElTimer aChrono;
   if (! EAMIsInit(&mNbMaxSel))
   {
        mNbMaxSel = mQuick ? QuickDefNbMaxSel : StdDefNbMaxSel;
   }
   mTKNbMaxSel = mQuick ? QuicTKNbMaxSel : StdTKNbMaxSel;

   if (mAotShow) 
      mQuitExist = false;


   mNbMaxInit = mQuick ? QuickNbMaxInit   : StdNbMaxInit ;


   if (! EAMIsInit(&mAotShow))
       mAotShow  = EAMIsInit(&mSzShow);


   if (MMVisualMode) return;
   

   cNewO_OneIm * mNoIm3 = new cNewO_OneIm(*mNM,m3S.mV2);

   std::string aNameSauveXml = mNM->NameOriOptimTriplet(false,mNoIm1,mNoIm2,mNoIm3,false);
   std::string aNameSauveBin = mNM->NameOriOptimTriplet(true ,mNoIm1,mNoIm2,mNoIm3,false);

   if (ELISE_fp::exist_file(aNameSauveXml) && ELISE_fp::exist_file(aNameSauveBin) && mQuitExist)
      return ;

   std::string  aName3R = mNM->NameOriInitTriplet(true,mNoIm1,mNoIm2,mNoIm3);
   cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);

   mInOriOk = false;

   if (EAMIsInit(&mInOri))
   {
       mInOriOk =    (mNM->CamOriOfNameSVP(mNoIm1->Name(),mInOri) !=0)
                  && (mNM->CamOriOfNameSVP(mNoIm2->Name(),mInOri) !=0)
                  && (mNM->CamOriOfNameSVP(mNoIm3->Name(),mInOri) !=0) ;
       // Si mInOriOk, apres NO_GenTripl, aXml3Ori contient les resultats de
       // InOri. Ensuite les rotation sont initialisees avec aXml3Ori.
       // Il suffit donc de bloquer la restimation (init et bundle) pour avoir
       // dans les "best sol" le resultat de InOri
   }

   


   mIms.push_back(new  cImOfTriplet(0,*this,mNoIm1,ElRotation3D::Id));
   mIms.push_back(new  cImOfTriplet(1,*this,mNoIm2,Xml2El(aXml3Ori.Ori2On1())));
   mIms.push_back(new  cImOfTriplet(2,*this,mNoIm3,Xml2El(aXml3Ori.Ori3On1())));

   mIm1 = mIms[0];
   mIm2 = mIms[1];
   mIm3 = mIms[2];
   mFoc =  1/ (  (1.0/mIm1->Foc()+1.0/mIm2->Foc()+1.0/mIm3->Foc()) / 3.0 ) ;


   // mP12 = new cPairOfTriplet(mIm1,mIm2,mIm3);
   // mP13 = new cPairOfTriplet(mIm1,mIm3,mIm2);
   // mP23 = new cPairOfTriplet(mIm2,mIm3,mIm1);
   mPairs.push_back(new cPairOfTriplet(mIm1,mIm2,mIm3));
   mPairs.push_back(new cPairOfTriplet(mIm1,mIm3,mIm2));
   mPairs.push_back(new cPairOfTriplet(mIm2,mIm3,mIm1));
   mP12 = mPairs[0];
   mP13 = mPairs[1];
   mP23 = mPairs[2];


   mNM->LoadTriplet(mIm1->Im(),mIm2->Im(),mIm3->Im(),&mIm1->VFullPtOf3(),&mIm2->VFullPtOf3(),&mIm3->VFullPtOf3());



   if (EAMIsInit(&mSzShow) && (!EAMIsInit(&mNbMaxSel)))
   {
      mNbMaxSel = ModeShowNbMaxSel;
   }

   if (mAotShow) 
      std::cout << "Time load " << aChrono.uval() << "\n";

   int aNbFull3 = mIm2->VFullPtOf3().size() ;
   int aNb3 = round_up(CoutAttenueTetaMax(aNbFull3,mNbMaxSel));
   mNbMaxSel = aNb3;
   mSel3 = IndPackReduit(mIm2->VFullPtOf3(),mNbMaxInit,mNbMaxSel);


   // aNb3 = round_up(CoutAttenueTetaMax(mIm2->VFullPtOf3().size() , mTKNbMaxSel));
   mTKNbMaxSel = ElMin(mTKNbMaxSel,aNbFull3);
   mTKSel3 = IndPackReduit(mIm2->VFullPtOf3(),ElMin(aNbFull3,5*mTKNbMaxSel),ElMin(aNbFull3,mTKNbMaxSel));




   for (int aK=0 ; aK<3 ; aK++)
   {
        mIms[aK]->SetReduce(mSel3,mTKSel3);
        mFullH123.push_back(&(mIms[aK]->VFullPtOf3()));
        mRedH123.push_back(&(mIms[aK]->VRedPtOf3()));
        mTK_H123.push_back(&(mIms[aK]->VTK_PtOf3()));
   }

   mP12->SetPackRed(false,mNbMaxInit,mNbMaxSel,mSel3,*mFullH123[1]);
   mP23->SetPackRed(true ,mNbMaxInit,mNbMaxSel,mSel3,*mFullH123[1]);
   mP13->SetPackRed(true ,mNbMaxInit,mNbMaxSel,mSel3,*mFullH123[0]);
   mPds3 = ElMin(MaxSurPond3,(mP12->NbR()+mP13->NbR()+mP23->NbR())/double(mIm1->NbR()));


   mBestResidu =  ResiduGlob();

   if (mAotShow) 
   {
      std::cout << "Time reduc " << aChrono.uval()   << "  Pds3=" << mPds3 << "\n";
   }

   if (!mInOriOk)
   {
      for (int aKP=0 ; aKP<int(mPairs.size()) ; aKP++)
      {
          TestOPA(*(mPairs[aKP]));
      }
   }

   if (mAotShow) 
   {
      std::cout << "Time opa " << aChrono.uval()   << "\n";
   }


   if (! mInOriOk)
      TestTomasiKanade();

   if (mAotShow)
   {
      std::cout << "NB TRIPLE " << mIm2->VFullPtOf3().size()  << " Resi3: " <<  ResiduTriplet() << " F " << mFoc << "\n";
      std::cout << "RESIDU/PAIRES " << mP12->ResiduMoy() << " " << mP13->ResiduMoy() << " " << mP23->ResiduMoy() << " " << "\n";
      std::cout << "R Glob " << ResiduGlob() << "\n";
   }

#if (ELISE_X11)

   if (EAMIsInit(&mSzShow))
   {
      mIm2->InitW(mSzShow);
      ShowPoints(mIm2,mP12->FullVP2(),P8COL::cyan,2);
      ShowPoints(mIm2,mP23->FullVP1(),P8COL::yellow,2);

      ShowPointSel(mIm2,mP12->VRedP2(),P8COL::cyan);
      ShowPointSel(mIm2,mP23->VRedP1(),P8COL::yellow);
      std::cout << "NB 12 " << mP12->VRedP2().size() << "\n";
      // ShowPoints(mIm2,mIm2->VFullPtOf3(),P8COL::blue,4);
      // ShowPointSel(mSel3.mVSel,mIm2,mIm2->VFullPtOf3(),P8COL::red);

      // 
      ShowPoints(mIm2,mIm2->VFullPtOf3(),P8COL::blue,2);
      ShowPointSel(mIm2,mIm2->VRedPtOf3(),P8COL::blue);


      //==================================
      mIm1->InitW(mSzShow);
      ShowPoints(mIm1,mP13->FullVP1(),P8COL::cyan,2);
      ShowPointSel(mIm1,mP13->VRedP1(),P8COL::cyan);

      ShowPoints(mIm1,mIm1->VFullPtOf3(),P8COL::blue,2);
      ShowPointSel(mIm1,mIm1->VRedPtOf3(),P8COL::blue);

      mIm2->W()->clik_in();
   }

#endif

   int aNbIterBundle = mQuick ? QuickNbIterBundle : StdNbIterBundle;
   if (mInOriOk) 
      aNbIterBundle =0;
   double aBOnH;
   Pt3dr aPMed;

   ElRotation3D aR21 =  mIm2->Ori();
   ElRotation3D aR31 =  mIm3->Ori();

   cParamCtrlSB3I aParamSB3I(aNbIterBundle);
   SolveBundle3Image
   (
        mFoc,
        aR21, // mIm2->Ori(),
        aR31, // mIm3->Ori(),
        aPMed,
        aBOnH,
        mRedH123,
        mP12->RedHoms(),
        mP13->RedHoms(),
        mP23->RedHoms(),
        mPds3,
        aParamSB3I
   );


   cXml_Ori3ImInit aXml;
   aXml.Ori2On1() = El2Xml(mIm2->Ori());
   aXml.Ori3On1() = El2Xml(mIm3->Ori());
   aXml.ResiduTriplet() = ResiduGlob();
   aXml.NbTriplet() = (int)mRedH123[0]->size();
   aXml.BSurH() = aBOnH;
   aXml.PMed() = aPMed;


   std::vector<Pt2df> & aVP1 = *(mRedH123[0]);
   std::vector<Pt2df> & aVP2 = *(mRedH123[1]);
   std::vector<Pt2df> & aVP3 = *(mRedH123[2]);

   //calcul d'ellipse par triplet
   cXml_Elips3D anElips3D;
   RazEllips(anElips3D);

   for (int aK=0; aK<int(aVP1.size()); aK++)
   {
       std::vector<Pt3dr> aW1;
       std::vector<Pt3dr> aW2;
       AddSegOfRot(aW1,aW2,mIm1->Ori(),aVP1.at(aK));
       AddSegOfRot(aW1,aW2,mIm2->Ori(),aVP2.at(aK));
       AddSegOfRot(aW1,aW2,mIm3->Ori(),aVP3.at(aK));

       bool OkI;
       Pt3dr aI = InterSeg(aW1,aW2,OkI);
        
       AddEllips(anElips3D,aI,1.0);
   }
   NormEllips(anElips3D);
   aXml.Elips() = anElips3D; 

   MakeFileXML(aXml,aNameSauveXml);
   MakeFileXML(aXml,aNameSauveBin);

   if (0)
       std::cout << anElips3D.CDG() << " " << anElips3D.Sxx() << " " << anElips3D.Syy() << " " << anElips3D.Szz() << "\n";

   if (mAotShow)
   {
      std::cout << "NB TRIPLE " << mIm2->VFullPtOf3().size()  << " Resi3: " <<  ResiduTriplet() << " F " << mFoc << "\n";
      std::cout << "RESIDU/PAIRES " << mP12->ResiduMoy() << " " << mP13->ResiduMoy() << " " << mP23->ResiduMoy() << " " << "\n";
      std::cout << "R Glob " << ResiduGlob() << "\n";
      std::cout << "Time bundle " << aChrono.uval() << "\n";
   }


    for (int aK=0 ; aK<3 ; aK++)
    {
        delete  mPairs[aK];
        delete  mIms[aK];
    }
    mPairs.clear();
    mIms.clear();
    mFullH123.clear();
    mRedH123.clear();
    mTK_H123.clear();
}


/**************************************************/
/*                                                */
/*            ::                                  */
/*                                                */
/**************************************************/

int CPP_OptimTriplet_main(int argc,char ** argv)
{
   cAppliOptimTriplet anAppli(argc,argv,true);

   return EXIT_SUCCESS;
}





int CPP_AllOptimTriplet_main(int argc,char ** argv)
{
   ElTimer aChrono;
   std::string aFullPat;
   bool inParal=true;
   bool Debug  = false;
   bool aShow = false;
   int  aNb0=0;
   cCommonMartiniAppli aCMA;

   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aFullPat,"Pattern"),
        LArgMain() 
                   << EAM(inParal,"Paral",true,"Execute in parallel ", eSAM_IsBool)
                   << EAM(Debug,"Debug",true,"Debugging mode (tuning purpose)", eSAM_IsBool)
                   << EAM(aShow,"Show",true,"Print command of each triplet")
                   << EAM(aNb0,"Nb0",true,"Num first pair to execute, tuning, Def=0")
                   << aCMA.ArgCMA()
    );

   cElemAppliSetFile anEASF(aFullPat);
   const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();

{
   std::cout << "aVImaVIm " << aVIm->size() << "\n";
}

   /// cSetName * aSet N= anEASF.mICNM->KeyOrPatSelector(aFullPat);
   std::set<std::string> aSetName(aVIm->begin(),aVIm->end());
   std::string aDir = anEASF.mDir;


   cNewO_NameManager * aNM =  new cNewO_NameManager(aCMA.mExtName,aCMA.mPrefHom,aCMA.mQuick,aDir,aCMA.mNameOriCalib,"dat");


   cSauvegardeNamedRel aLCpl =  StdGetFromPCP(aNM->NameCpleOfTopoTriplet(true),SauvegardeNamedRel);
   std::list<std::string> aLCom;
   int aNb= 0 ;
   int aNb2 = (int)aLCpl.Cple().size();
   for (std::vector<cCpleString>::const_iterator itC=aLCpl.Cple().begin() ; itC!=aLCpl.Cple().end() ; itC++)
   {
       aNb++;
       const std::string & aN1 = itC->N1();
       const std::string & aN2 = itC->N2();
       // if (aSet N->SetBasicIsIn(aN1) && aSet N->SetBasicIsIn(aN2) && (aNb>=(aNb0+1)))
       if (DicBoolFind(aSetName,aN1) && DicBoolFind(aSetName,aN2) && (aNb>=(aNb0+1)))
       {
            std::string aCom =   MM3dBinFile("TestLib NO_OneImOptTrip") 
                            + " " + aN1
                            + " " + aN2
                            + " " + cAppliOptimTriplet::KeyCple;
             aCom += aCMA.ComParam();
/*
            if (EAMIsInit(&aNameCalib))
               aCom +=  " OriCalib=" + aNameCalib;

            aCom += " Quick=" + ToString(Quick);
            aCom += " SH=" + aPrefHom;
            aCom += " ExtName=" + aExtName;
            aCom += " ModeNO=" + aNameModeNO;
            aCom += " InOri=" + aInOri;
*/

            if (aShow)
            {
               std::cout << "#COM=[" << aCom << "]\n";
            }
            else
            {
                if (inParal)
                {
                    aLCom.push_back(aCom);
                    if ((aNb%40) == 0)
                    {
                        cEl_GPAO::DoComInParal(aLCom);
                        aLCom.clear();
                        std::cout << "Optim triplets Done " << aNb << " pairs out of " << aNb2  << " in " << aChrono.uval() << "\n";
                    }
                }
                else
                {
                    aCom += " CalledBySerie=true";
                    std::cout << "COM " << aCom << "\n";
                    System(aCom);
                }
            }
       }
   }
   if (! aShow)
      cEl_GPAO::DoComInParal(aLCom);

   return EXIT_SUCCESS;
}




/*
    //  VERSION AVEC TRIPLETS - A CONSERVER AU CAS OU
    // REMPLACEE PAR UNE VERSION AVEC 1/Image par couple
int CPP _AllOptimTriplet_main(int argc,char ** argv)
{
   ElTimer aChrono;
   std::string aFullPat,aNameCalib;
   bool inParal=true;
   bool Quick = false;

   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aFullPat,"Pattern"),
        LArgMain() << EAM(aNameCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(inParal,"Paral",true,"Execute in parallel ", eSAM_IsBool)
                   << EAM(Quick,"Quick",true,"Quick version", eSAM_IsBool)
    );

   cElemAppliSetFile anEASF(aFullPat);
   const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();
   std::set<std::string> aSetName(aVIm->begin(),aVIm->end());
   std::string aDir = anEASF.mDir;

   cNewO_NameManager * aNM =  new cNewO_NameManager(Quick,aDir,aNameCalib,"dat");
   cXml_TopoTriplet aXml3 =  StdGetFromSI(aNM->NameTopoTriplet(true),Xml_TopoTriplet);
   int aNb3 = aXml3.Triplets().size();
   std::list<std::string> aLCom;

   int aNb= 0 ;
   for 
   (
      std::list<cXml_OneTriplet>::const_iterator it3=aXml3.Triplets().begin() ;
      it3 !=aXml3.Triplets().end() ;
      it3++
   )
   {
         aNb++;
         if (
                    DicBoolFind(aSetName,it3->Name1())
                &&  DicBoolFind(aSetName,it3->Name2())
                &&  DicBoolFind(aSetName,it3->Name3())
            )
         {
            std::string aCom =   MM3dBinFile("TestLib NO_OneImOptTrip") 
                            + " " + it3->Name1() 
                            + " " + it3->Name2() 
                            + " " + it3->Name3()  ;
            if (EAMIsInit(&aNameCalib))
               aCom +=  " OriCalib=" + aNameCalib;

            aCom += " Quick=" + ToString(Quick);
            if (inParal)
            {
                aLCom.push_back(aCom);
                if ((aNb%100) == 0)
                {
                    cEl_GPAO::DoComInParal(aLCom);
                    aLCom.clear();
                    std::cout << "Optim triplets Done " << aNb << " out of " << aNb3 << "\n";
                }
            }
            else
            {
                std::cout << "COM " << aCom << "\n";
                System(aCom);
            }
         }

   }
   cEl_GPAO::DoComInParal(aLCom);
 
   
   return EXIT_SUCCESS;
}
*/






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
