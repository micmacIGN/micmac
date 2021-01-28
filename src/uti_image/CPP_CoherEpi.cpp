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
//#include "StdAfx.h"
#include "QualDepthMap.h"


#if (0)
#endif

extern Im2D_Bits<1>  TestLabel(Im2D_INT2 aILabel,INT aLabelOut);
extern Fonc_Num  MasqBorHomogene(Im2D_REAL4 anIm0,Im2D_Bits<1>  aMasq0,Video_Win * aW);


bool DEBUGM3D = false;
bool DEB_NU3D = false;

/*******************************************************************/
/*                                                                 */
/*                cCEM_OneIm_Epip                                  */
/*                                                                 */
/*******************************************************************/


cCEM_OneIm_Epip::cCEM_OneIm_Epip (cCoherEpi_main * aCEM,const std::string & aName,const Box2di & aBox,bool aVisu,bool IsFirstIm,bool Final) :
   cCEM_OneIm(aCEM,aName,aBox,aVisu,IsFirstIm),
   mNamePx    (mDir+ (   mCple ?
                         mCple->LocPxFileMatch(mNameInit,mCoher->mNumPx,mCoher->mDeZoom) :
                         LocPxFileMatch(mDirM,mCoher->mNumPx,mCoher->mDeZoom)
                     )
              ),
   mTifPx     (mNamePx.c_str()),
   mImPx      (mSz.x,mSz.y),
   mTPx       (mImPx),
   mImPx_u2   (mSz.x,mSz.y),
   mNameMasq  (mDir+ (   mCple ?
                         mCple->LocMasqFileMatch(mNameInit,mCoher->mNumMasq):
                         LocMasqFileMatch(mDirM,mCoher->mNumMasq)
                     )
              ),
   mTifMasq   (mNameMasq.c_str()),
   mCam       (0)
{
    if (Empty()) return;

    if ((!DEBUGM3D) && type_im_integral(mTifPx.type_el()))
    {
        Im2D_INT2 anIQ(mSz.x,mSz.y);
        ELISE_COPY ( mIm.all_pts(),trans(mTifPx.in_proj(),mP0) ,anIQ.out());
        ElImplemDequantifier aDeq(mSz);
        aDeq.DoDequantif(mSz,anIQ.in());
        ELISE_COPY(mImPx.all_pts(),aDeq.ImDeqReelle()*mCoher->mStep,mImPx.out());
    }
    else
    {
        ELISE_COPY (mImPx.all_pts(),trans(mTifPx.in_proj(),mP0) * mCoher->mStep ,mImPx.out());
    }

    if (Final)
    {
        std::string aNamePxU2 = mDir+   (  mCple ?
                                           mCple->LocPxFileMatch(mNameInit,mCoher->mNumPx-1,mCoher->mDeZoom) :
                                           LocPxFileMatch(mDirM,mCoher->mNumPx-1,mCoher->mDeZoom)
                                        );
        Tiff_Im aTpxU2 (aNamePxU2.c_str());
        ELISE_COPY(mImPx_u2.all_pts(),trans(aTpxU2.in_proj(),mP0),mImPx_u2.out());

    }
    ELISE_COPY(mImMasq.all_pts(),trans(mTifMasq.in(0),mP0),mImMasq.out());

    if (aCEM->Masq3d())
    {
        ELISE_ASSERT(mCple,"Masq3 require orientaion !!");
        std::string aNameCam =   aCEM->ICNM()->Dir()
                               + aCEM->ICNM()->Assoc1To1("NKS-Assoc-Im2Orient@-Epi",mNameImMatched,true);
        mCam = CamOrientGenFromFile(aNameCam,aCEM->ICNM());
    }

    if (DEB_NU3D)
    {
       mNNOri =  "NuageImProf_Chantier-Ori_Etape_"+ToString(mCoher->mNumPx) + ".xml";
       mNuOri = cElNuage3DMaille::FromFileIm(mDirM+mNNOri);
       std::cout << "KKKKKK  " << mNamePx << " " << mNameMasq << "\n";
       std::cout << "DDDDD  " << mDirM << "\n\n";
    }
}

void cCEM_OneIm_Epip::UsePack(const ElPackHomologue & aPack)
{
     for (ElPackHomologue::const_iterator itH = aPack.begin() ;  itH!=aPack.end(); itH++)
     {
         Pt2dr aP1 = itH->P1();
         Pt2dr aP2 = itH->P2();
         bool Ok;
         Pt2dr aQ2 = ToIm2(aP1,Ok);

         //std::cout << aQ2 - aP2 <<  " " << mTPx.getr(aP1,0) << "\n";

         double aD = ElAbs(aQ2.x-aP2.x);
         if (0) mWin->draw_circle_abs(aP1,3.0,mWin->pdisc()( (aD<0.5) ? P8COL::green : P8COL::red));
     }
}


Pt3dr  cCEM_OneIm_Epip::To3d(const Pt2di & anI,const Pt3dr *) const
{
    cCEM_OneIm_Epip * aE2 = static_cast<cCEM_OneIm_Epip*> (mConj);
    CamStenope * aCam2 = aE2->mCam;
    Pt2dr  aP1 = ToImCam(Pt2dr(anI));
    Pt2dr  aP2 =  mConj->ToImCam(ToIm2Gen(Pt2dr(anI)));
    // Pt2dr  aP2 = ( Pt2dr(anI) + Pt2dr(mTPx.get(anI),0)) * mCoher->mDeZoom;
    // Pt2dr  aP2 = aP1 +  Pt2dr((mTPx.get(anI)/mCoher->mStep )*mResolAlti,0);
    // aP2 = aP1 +  Pt2dr(mTPx.get(anI)*mResolAlti*2,0);
// std::cout << mCoher->mStep << " " << mResolAlti << "\n";


    return mCam->PseudoInter(aP1,*aCam2,aP2);
    // ELISE_ASSERT(false,"cCEM_OneIm_Epip::To3d");
    // return Pt3dr(0,0,0);
}


/*******************************************************************/
/*                                                                 */
/*                cCEM_OneIm_Nuage                                 */
/*                                                                 */
/*******************************************************************/

cCEM_OneIm_Nuage::cCEM_OneIm_Nuage(cCoherEpi_main * aCoh,const std::string & aName1,const std::string & aName2,const Box2di & aBox,bool Visu,bool IsFirstIm) :
    cCEM_OneIm  (aCoh,aName1,aBox,Visu,IsFirstIm),
    mDirLoc1    (LocDirMec2Im(aName1,aName2)),
    mDirLoc2    (LocDirMec2Im(aName2,aName1)),
    mDirNuage1  (mDir+mDirLoc1),
    mDirNuage2  (mDir+mDirLoc2),
    mPGMN       (1.0,I2R(aBox),true),
    mNameN      ("NuageImProf_Chantier-Ori_Etape_" +ToString(mCoher->mNumPx) + ".xml"),
    mParam1     (StdGetFromSI(mDirNuage1+mNameN,XML_ParamNuage3DMaille)),
    mNuage1     (cElNuage3DMaille::FromParam(aName1,mParam1,mDirNuage1,"",1.0,&mPGMN,false)),
    mParam2     (StdGetFromSI(mDirNuage2+mNameN,XML_ParamNuage3DMaille)),
    mNuage2     (cElNuage3DMaille::FromParam(aName2,mParam2,mDirNuage2,"",1.0,(const cParamModifGeomMTDNuage *)0,true))
{
    ELISE_COPY
    (
       mNuage1->ImDef().all_pts(),
       mNuage1->ImDef().in(),
       mImMasq.out()
    );
}


Pt3dr cCEM_OneIm_Nuage::To3d(const Pt2di & anI,const Pt3dr *) const
{
   return mNuage1->PtOfIndex(anI);
}

          // cElNuage3DMaille *       mNuage;
/*******************************************************************/
/*                                                                 */
/*                cCEM_OneIm                                       */
/*                                                                 */
/*******************************************************************/

/*
static Box2di  ShowBox(const Box2di & aBox)
{
    std::cout << "BBBB=" << aBox._p0 << " " << aBox._p1 << "\n";
    return aBox;
}
static Pt2di  ShowPt(const Pt2di & aPt)
{
    std::cout << "PPPPP=" << aPt << "\n";
    return aPt;
}
*/

cCEM_OneIm::cCEM_OneIm
(
    cCoherEpi_main *       aCoher,
    const std::string &    aName,
    const Box2di      &    aBox,
    bool                   aVisu,
    bool                   IsFirstIm
)  :
   mIsFirstIm (IsFirstIm),
   mCoher     (aCoher),
   mCple      (mCoher->mCple),
   mDir       (mCoher->mDir),
   mDirM      (mDir + (mCple ? mCple->LocDirMatch(aName) : LocDirMec2Im(aCoher->NameIm(IsFirstIm),aCoher->NameIm(!IsFirstIm)))),
   mNameNuage ("NuageImProf_LeChantier_Etape_"+ToString(mCoher->mNumPx) + ".xml"),
   mParNuage  (StdGetFromSI(mDirM+mNameNuage,XML_ParamNuage3DMaille)),
   mResolAlti (mParNuage.Image_Profondeur().Val().ResolutionAlti()),
   mNameInit  (aName),
   mNameImMatched (mCple ? mCple->LocNameImEpi(mNameInit) : mNameInit),
   mNameFinal (mDir+  (mCple ? mCple->LocNameImEpi(mNameInit,mCoher->mDeZoom,false) : StdNameImDeZoom(mNameInit,mCoher->mDeZoom))),
   mTifIm     (Tiff_Im::UnivConvStd(mNameFinal.c_str())),
   mBox       (Inf(aBox,Box2di(Pt2di(0,0),mTifIm.sz()))),
   mSz        (mBox.sz()),
   mP0        (mBox._p0),
   mRP0       (mP0),
   mIm        (mSz.x,mSz.y),
   mImOrtho   (1,1),
   mWin       ((aVisu && IsFirstIm) ? Video_Win::PtrWStd(mSz) : 0),
   mWin2      (0),
   mConj      (0),
   mImMasq    (mSz.x,mSz.y,1),
   mTMasq     (mImMasq),
   mNNOri     (""),
   mNuOri     (0)
{
    ELISE_COPY
    (
         mIm.all_pts(),
         trans(mTifIm.in(),mP0),
         mIm.out()  | VMax(mVMaxIm)
    );
    if (mWin)
    {
       double aVMax;
       ELISE_COPY(mIm.all_pts(),mIm.in(),VMax(aVMax));
       ELISE_COPY ( mIm.all_pts(), 255.0* Min(mIm.in()/aVMax,1.0), VGray());
       mWin->set_title(IsFirstIm ? "Image 1" : "Image 2");
    }

}



void cCEM_OneIm::PostInit()
{

    if (mCoher->Masq3d())
    {

        std::vector<Pt3dr> aVPts;
        std::vector<Pt3di> aVCoul;

        std::vector<Pt3dr> aVPtsNu;
        std::vector<Pt3di> aVCoulNu;

        int aCptIn= 0;
        Pt2di anI;
        for (anI.x =0 ; anI.x <mSz.x ; anI.x++)
        {
            for (anI.y =0 ; anI.y <mSz.y ; anI.y++)
            {
                if (mTMasq.get(anI))
                {
                   aCptIn++;
                   if (DEBUGM3D)
                   {
                      Pt3dr aPt = To3d(anI,0);
                      bool InMasq = mCoher->Masq3d()->IsInMasq(aPt);
                      aVPts.push_back(aPt);
                      aVCoul.push_back(InMasq ? Pt3di(128,128,128) : Pt3di(255,0,0));
                   }

                   if (DEB_NU3D)
                   {
                      Pt2di  anI0 = anI + mP0;
                      if (mNuOri->IndexHasContenu(anI0))
                      {
                          Pt3dr aPtNu = mNuOri->PtOfIndex(anI0);

                          Pt3dr aPt = To3d(anI,0);
                          std::cout << "Ddddd " << euclid(aPtNu-aPt) << "\n";

                          bool InMasqNu = mCoher->Masq3d()->IsInMasq(aPtNu);
                          aVPtsNu.push_back(aPtNu);
                          aVCoulNu.push_back(InMasqNu ? Pt3di(0,255,0) : Pt3di(0,0,255));
                      }
                   }
                   if ((!DEBUGM3D) && (!DEB_NU3D))
                   {
                      if  (! mCoher->Masq3d()->IsInMasq(To3d(anI,0)))
                          mTMasq.oset(anI,0);
                   }
                }
            }
        }
        std::string aPref = mIsFirstIm ? "-FIRST" : "-SECOND" ;
        if (DEBUGM3D)
        {
           std::list<std::string> aVCom;
           std::vector<const cElNuage3DMaille *> aVNuage;

           cElNuage3DMaille::PlyPutFile
           (
              mDirM  + StdPrefix(mNameNuage) +  aPref + "-TestM3D.ply",
              aVCom,
              aVNuage,
              &(aVPts),
              &(aVCoul),
              true
           );
        }
        if (DEBUGM3D)
        {
           std::list<std::string> aVCom;
           std::vector<const cElNuage3DMaille *> aVNuage;

           std::string aName = mDirM  + StdPrefix(mNNOri) +  aPref + "-TM3D.ply";

           cElNuage3DMaille::PlyPutFile
           (
              aName,
              aVCom,
              aVNuage,
              &(aVPtsNu),
              &(aVCoulNu),
              true
           );
           std::cout << "WWWAAIIt " << mDirM << " " << mNNOri << " " << aCptIn << "\n";
           std::cout << aName << "\n";

           getchar();
        }
    }



}

bool cCEM_OneIm::Empty() const
{
    return (mSz.x==0) || (mSz.y==0);
}

void cCEM_OneIm::UsePack(const ElPackHomologue &)
{
}


void cCEM_OneIm::SetConj(cCEM_OneIm * aConj)
{
   mConj = aConj;
   mConj->mConj = this;
}

Box2dr cCEM_OneIm::BoxIm2(const Pt2di & aSzIm2,bool & OneOk)
{
   Pt2dr aP0(1e9,1e9);
   Pt2dr aP1(-1e9,-1e9);
   OneOk = false;

   Pt2di aPIm;
   for (aPIm.x=0 ; aPIm.x<mSz.x ; aPIm.x++)
   {
       for (aPIm.y=0 ; aPIm.y<mSz.y ; aPIm.y++)
       {
          // if (mTMasq.get(aPIm))
          if (IsOK(aPIm))
          {
             bool Ok;
             Pt2dr aPIm2 = RoughToIm2(Pt2dr(aPIm),Ok) ;
             if (Ok)
             {
                aP0.SetInf(aPIm2);
                aP1.SetSup(aPIm2);
                OneOk = true;
             }
          }
       }
   }
   if (!OneOk)
      return I2R(Inf(mBox,Box2di(Pt2di(0,0),aSzIm2)));

   return Box2dr(aP0,aP1);
}
Box2dr cCEM_OneIm::BoxIm2(const Pt2di & aSzIm2)
{
   bool OneOk;
   return BoxIm2(aSzIm2,OneOk);
}

static Pt2di PBUG(180,418);
Im2D_U_INT1  cCEM_OneIm::ImAR()
{
   Im2D_U_INT1 aRes(mSz.x,mSz.y,0);
   TIm2D<U_INT1,INT> aTRes(aRes);

   double aSigma = mCoher->mSigmaP;
   Pt2di aPIm;
   for (aPIm.x=0 ; aPIm.x<mSz.x ; aPIm.x++)
   {
       for (aPIm.y=0 ; aPIm.y<mSz.y ; aPIm.y++)
       {
           bool  Ok;
           Pt2dr aQ = AllerRetour(Pt2dr(aPIm),Ok);
           if (0) // (aPIm==PBUG)
           {
              std::cout << "OOkkkkk " << Ok << "\n";
              Pt2dr aP2 =  ToIm2(Pt2dr(aPIm),Ok);
              std::cout << "OOkkkkk2 " << Ok << aP2 << "\n";
              std::cout << "OOkkkkk3 " <<  mConj->IsOK(round_ni(aP2)) << "\n";
           }
           if (Ok)
           {
               double aRatio = euclid(Pt2dr(aPIm)-aQ) / aSigma;
               double aPds = 255/(1+aRatio);
               aTRes.oset(aPIm,ElMin(255,round_ni(aPds)));
           }
       }
   }
   if (0)
   {
      ELISE_COPY(aRes.all_pts(),aRes.in(),VGray());
      getchar();
   }


   return aRes;
}

Im2D_REAL4  cCEM_OneIm::VerifProf(Im2D_Bits<1> aMasq)
{

    Im2D_REAL4 anIQG = ImageQualityGrad(ImPx(),aMasq,mWin);

    if (0) // (mWin)
    {
       ELISE_COPY(anIQG.all_pts(),Min(255,100*anIQG.in()),mWin->ogray());

       mWin->clik_in();
    }

    return anIQG;
}

Fonc_Num Grad(Fonc_Num aF)
{
  return aF -trans(aF,Pt2di(1,0));
}

void cCEM_OneIm::VerifIm(Im2D_Bits<1> aMasq)
{
   if (Empty()) return;

   ComputeOrtho();

   Im2D_U_INT1 aIMin(mSz.x,mSz.y,255);

  for (int aSzW = 1; aSzW<2 ; aSzW++)
  {

      // Symb_FNum  aSymbM(aMasq.in_proj());
      Symb_FNum  aSymbM(1);
      Symb_FNum  aSymbF1(Grad(mIm.in_proj()));
      Symb_FNum  aSymbF2(Grad(mImOrtho.in_proj()));

      Symb_FNum  aSFoncs (Virgule(1,aSymbF1,aSymbF2,Square(aSymbF1),Square(aSymbF2),aSymbF1*aSymbF2));
      Symb_FNum  aSomFoncs(rect_som(aSFoncs,aSzW));

      Symb_FNum aS0  ( aSomFoncs.kth_proj(0));
      Symb_FNum aS1  ( aSomFoncs.kth_proj(1) /aS0);
      Symb_FNum aS2  ( aSomFoncs.kth_proj(2) /aS0);
      Symb_FNum aS11 ( aSomFoncs.kth_proj(3) /aS0 - Square(aS1));
      Symb_FNum aS22 ( aSomFoncs.kth_proj(4) /aS0 - Square(aS2));
      Symb_FNum aS12 ( aSomFoncs.kth_proj(5) /aS0 - aS1*aS2);

      Symb_FNum aCor = aS12 / sqrt(Max(1e-2,aS11*aS22));

       ELISE_COPY
       (
            mImOrtho.all_pts(), Max(0,Min(aIMin.in(), 255*(aCor>0.5))) , mWin2->ogray() | aIMin.out()
       );
   }
}

void cCEM_OneIm::ComputeOrtho()
{

   mImOrtho.Resize(mIm.sz());
   Pt2di aSz =  mImOrtho.sz();
   TIm2D<REAL4,REAL8> aTIm(mIm);
   TIm2D<REAL4,REAL8> aTImOr(mImOrtho);
   TIm2D<REAL4,REAL8> aTIm2(mConj->mIm);

   Pt2di aP;

   for (aP.x=0 ; aP.x<aSz.x; aP.x++)
   {
       for (aP.y=0 ; aP.y<aSz.y; aP.y++)
       {
           // bool Ok;
           // Pt2dr  aP2 = ToIm2(Pt2dr(aP),Ok);

           Pt2dr  aP2 = ToIm2Gen(Pt2dr(aP));
           double aVal =  aTIm2.getr(aP2,0.0);
           aTImOr.oset(aP,aVal);
       }
   }
   ELISE_COPY(mImOrtho.all_pts(), mImOrtho.in(),VMax(mVMaxOrtho));

   if (mWin)
   {
       mWin2  =  Video_Win::PtrWStd(mSz) ;
       mWin2->set_title("Image Ortho");
       ELISE_COPY (  mImOrtho.all_pts(), 255.0* Min( mImOrtho.in()/mVMaxOrtho,1.0), mWin2->ogray());
   }
}

/*******************************************************************/
/*                                                                 */
/*                cCoherEpi_main                                   */
/*                                                                 */
/*******************************************************************/


cCoherEpi_main::cCoherEpi_main (int argc,char ** argv) :
    mSzDecoup (2000),
    mBrd      (250),
    mDir      ("./"),
    mMasq3d   (0),
    mCple     (0),
    mWithEpi  (true),
    mNoOri    (false),
    mByP      (true),
    mInParal  (true),
    mCalledByP(false),
    mPrefix    ("AR"),
    mPostfixP  ("_Glob"),
    mDeZoom   (1),
    mNumPx    (9),
    mNumMasq  (8),
    mVisu     (false),
    mSigmaP   (1.5),
    mRegulCheck (true),
    mFilterBH   (true),
    mStep     (1.0),
    mRegul    (0.5),
    mReduceM  (2.0),
    mFinal    (false),
    mImQualDepth (1,1),
    mDoMasq      (false),
    mDoMasqSym   (false),
    mUseAutoMasq   (true),
    mBSHRejet      (0.02)

{
    std::string aNameMasq3D;
    bool Debug=false;
    double aFactBSHOk=2;
    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(mNameIm1,"Name Im1", eSAM_IsExistFile)
                    << EAMC(mNameIm2,"Name Im2", eSAM_IsExistFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistDirOri),
    LArgMain()  << EAM(mDir,"Dir",true)
                    << EAM(mBoxIm1,"Box",true)
                    << EAM(mSzDecoup,"SzDec",true)
                    << EAM(mBrd,"Brd",true)
                    << EAM(mSigmaP,"SigP",true,"Standard error in pixel (Def=1.5)")
                    << EAM(mRegulCheck,"RegCh",true,"Check regularity of DSM (avoid with vegetation)")
                    << EAM(mFilterBH,"FBH",true,"Filter on homogeneous border (to avoid sky background)")
                    << EAM(mIntY1,"YBox",true)
                    << EAM(mRegul,"Regul",true,"Regularisation for mask (Def = 0.5)")
                    << EAM(mReduceM,"RedM",true,"Reduce factor for mask (Def = 2.0)")
                    << EAM(mDoMasqSym,"DoMS",true,"Do symetric mask (Def  = false)")
                    << EAM(mDoMasq,"DoM",true,"Do Mask, def = false")
                    << EAM(mUseAutoMasq,"UAM",true,"Use Auto Mask (def = same as DoM)")
                    << EAM(mVisu,"Visu",true)
                    << EAM(mDeZoom,"Zoom",true)
                    << EAM(mNumPx,"NumPx",true)
                    << EAM(mNumMasq,"NumMasq",true)
                    << EAM(mPrefix,"Prefix",true,"Prefix to result name, Def= AR")
                    << EAM(mStep,"Step",true)
                    << EAM(mWithEpi,"ByE",true)
                    << EAM(mByP,"ByP",true)
                    << EAM(mInParal,"InParal",true,"Run command in parallel, Def=true, tuning")
                    << EAM(mCalledByP,"InternalCalledByP",true)
                    << EAM(mPostfixP,"InternalPostfixP",true)
                    << EAM(mFinal,"ExpFin",true,"For final export (generate distortion indicator)")
                    << EAM(mBSHRejet,"BSHReject",true,"Value for low Basr to Ratio leading do rejection (Def=0.02)")
                    << EAM(aFactBSHOk,"FactBSHOk",true,"Multiplier so that BSHOk= FactBSHOk * BSHReject (Def=2)")
                    << EAM(Debug,"Debug",true,"Tuning ....", eSAM_InternalUse)
                    << EAM(aNameMasq3D,"Masq3D",true,"3D masq to enhance filtering")
   );

    if (MMVisualMode) return;

   if (EAMIsInit(&aNameMasq3D)) mMasq3d = cMasqBin3D::FromSaisieMasq3d(mDir + aNameMasq3D);
   mNoOri  =  (mOri  == "NONE");

   mBSHOk  = mBSHRejet * aFactBSHOk;

   if (mVisu)
   {
        mDoMasq = true;
        mByP = true;
   }

   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   if (! EAMIsInit(&mPrefix))
     mPrefix = mPrefix + mNameIm1 + "-" + mNameIm2 ;

/*
   ELISE_ASSERT
   (
        mNameIm1 < mNameIm2,
        "Image names must be ordered in CoherEpip"
   );
*/

   // if (! EAMIsInit(&mUseAutoMasq)) mUseAutoMasq = mDoMasq;
   if (! EAMIsInit(&mNumMasq))
   {
        mNumMasq =  (mDeZoom==1) ? (mNumPx-1) : mNumPx;
   }

   if (mWithEpi && (! mNoOri))
   {
      mCple = StdCpleEpip(mDir,mOri,mNameIm1,mNameIm2);
   }

   std::string aNameIm1DeZoom =  mCple                                          ?
                                 (mDir+ mCple->LocNameImEpi(mNameIm1,mDeZoom,false))  :
                                 StdNameImDeZoom(mNameIm1,mDeZoom)              ;
   std::string aNameIm2DeZoom =  mCple                                          ?
                                 (mDir+ mCple->LocNameImEpi(mNameIm2,mDeZoom,false))  :
                                 StdNameImDeZoom(mNameIm2,mDeZoom)              ;

   std::string aNameIm1Match = mCple ? mCple->LocNameImEpi(mNameIm1) : mNameIm1;
   std::string aNameIm2Match = mCple ? mCple->LocNameImEpi(mNameIm2) : mNameIm2;

   bool HasHom = false;
   ElPackHomologue  aPackH ;
   if (mCple)
   {
       // HasHom = true;  Pas utilise et bug sur couple sans homologue
       // std::string aNameH = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@@dat",aNameIm1Match,aNameIm2Match,true);
       // aPackH = ElPackHomologue::FromFile(aNameH);
   }


   Tiff_Im aTF1(aNameIm1DeZoom.c_str());
   Tiff_Im aTF2(aNameIm2DeZoom.c_str());
   Pt2di aSzIm2(aTF2.sz());


   Pt2di aSz1 = aTF1.sz() ;
   if (!EAMIsInit(&mBoxIm1))
   {
      if (EAMIsInit(&mIntY1))
      {
          mBoxIm1 = Box2di(Pt2di(0,mIntY1.x),Pt2di(aSz1.x,mIntY1.y));
      }
      else
      {
          mBoxIm1 = Box2di(Pt2di(0,0),aSz1);
      }
   }



   if (mByP && (!mCalledByP))
   {
   // Cas on lance le process en paral et onr recolle les morceaux
         std::string aCom = MMBinFile(MM3DStr) +  MakeStrFromArgcARgv(argc,argv);
         aCom = aCom + " InternalCalledByP=true";
         std::cout << "COM = " << aCom << "\n";
         Pt2di aPSzDecoup(mSzDecoup,mSzDecoup);
         Pt2di aPBrd(mBrd,mBrd);

         cDecoupageInterv2D aDecoup (mBoxIm1,aPSzDecoup,Box2di(-aPBrd,aPBrd));

         std::list<std::string> aLCom;
         std::vector<cBoxCoher> aVBoxC;

         for (int aKB=0 ; aKB<aDecoup.NbInterv() ; aKB++)
         {
             std::string aPost = "_BoxCoherEpip" + ToString(aKB);
             std::string aComBox = aCom + " Box=" + ToString(aDecoup.KthIntervIn(aKB))
                                        + " InternalPostfixP=" + aPost;
             // System(aComBox);
             aLCom.push_back(aComBox);
             if (Debug) std::cout << aComBox << "\n";
             aVBoxC.push_back(cBoxCoher(aDecoup.KthIntervIn(aKB),aDecoup.KthIntervOut(aKB),aPost));
         }
         if (mInParal)
         {
             cEl_GPAO::DoComInParal(aLCom,"MakeBascule");
         }
         else
         {
             cEl_GPAO::DoComInSerie(aLCom);
         }

         std::string aNameGlob = mDir+ mPrefix + mPostfixP + ".tif";
         std::string aNameMasqGlob1 = mDir+ mPrefix +"_Masq1" + mPostfixP + ".tif";
         std::string aNameMasqGlob2 = mDir+ mPrefix +"_Masq2" + mPostfixP + ".tif";
         Tiff_Im aTifGlob = Tiff_Im::Create8BFromFonc(aNameGlob,mBoxIm1.sz(),0);

         std::string aNameImDist = mDir+ mPrefix + "_ImDistor"  +mPostfixP + ".tif";

         Tiff_Im aTifMasq1 = aTifGlob;
         Tiff_Im aTifMasq2 = aTifGlob;
         Tiff_Im aTifDist = aTifGlob;
         if (mDoMasq)
         {
             aTifMasq1 = Tiff_Im(aNameMasqGlob1.c_str(),mBoxIm1.sz(),GenIm::bits1_msbf,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
             if (mDoMasqSym)
                 aTifMasq2 = Tiff_Im(aNameMasqGlob2.c_str(),aSzIm2,GenIm::bits1_msbf,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
         }
         if (mFinal)
         {
             aTifDist = Tiff_Im(aNameImDist.c_str(),mBoxIm1.sz(),GenIm::u_int1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
         }

         for (int aKB=0 ; aKB<int(aVBoxC.size()) ; aKB++)
         {
             std::string aNameC = mDir+ mPrefix+aVBoxC[aKB].mPost + ".tif";
             std::string aNameM1 = mDir+ mPrefix+ "_Masq1"+ aVBoxC[aKB].mPost + ".tif";
             std::string aNameM2 = mDir+ mPrefix+ "_Masq2"+ aVBoxC[aKB].mPost + ".tif";
             Box2di aBoxOut = aVBoxC[aKB].mBoxOut;
             Box2di aBoxIn = aVBoxC[aKB].mBoxIn;
             ELISE_COPY
             (
                rectangle(aBoxOut._p0,aBoxOut._p1),
                trans(Tiff_Im::StdConv(aNameC).in(),-aBoxIn._p0),
                aTifGlob.out()
             );
             ELISE_fp::RmFile(aNameC);

             if (mFinal)
             {
                 std::string aNamDistLoc =  mDir+ mPrefix + "_ImDistor" + aVBoxC[aKB].mPost + ".tif";
                 ELISE_COPY
                 (
                    rectangle(aBoxOut._p0,aBoxOut._p1),
                    trans(Tiff_Im::StdConv(aNamDistLoc).in(),-aBoxIn._p0),
                    aTifDist.out()
                 );
                 ELISE_fp::RmFile(aNamDistLoc);
             }
             if (mDoMasq)
             {
                 ELISE_COPY
                 (
                    rectangle(aBoxOut._p0,aBoxOut._p1),
                    trans(Tiff_Im::StdConv(aNameM1).in(),-aBoxIn._p0),
                    aTifMasq1.out()
                 );
                 if (! Debug)
                    ELISE_fp::RmFile(aNameM1);
                 if (mDoMasqSym)
                 {
                    std::string aNameFOM2 = mDir+mPrefix + "_DEC2"+ aVBoxC[aKB].mPost + ".xml";
                    cMTDCoher aMTD = StdGetFromPCP(aNameFOM2,MTDCoher);

                    Tiff_Im aTifLoc2(aNameM2.c_str());

                    ELISE_COPY
                    (
                         rectangle(aMTD.Dec2(), aMTD.Dec2()+aTifLoc2.sz()),
                         Max(trans(aTifLoc2.in(),-aMTD.Dec2()),aTifMasq2.in()),
                         aTifMasq2.out()
                    );

                    ELISE_fp::RmFile(aNameM2);
                    ELISE_fp::RmFile(aNameFOM2);
                 }
             }
         }

   }
   else
   {


   // Cas on fait le calcul

       if (mWithEpi)
          mIm1 = new cCEM_OneIm_Epip(this,mNameIm1,mBoxIm1,mVisu,true,mFinal)          ;
       else
          mIm1 = new cCEM_OneIm_Nuage(this,mNameIm1,mNameIm2,mBoxIm1,mVisu,true);


       bool  OneOk2;
       Box2di aBoxIm2 = R2ISup(mIm1->BoxIm2(aSzIm2,OneOk2));
       if (MPD_MM()) std::cout << "OneOk : " << OneOk2 << "\n";

       if (OneOk2)
       {
           if (mWithEpi)
           {
              mPIm2 = new cCEM_OneIm_Epip(this,mNameIm2,aBoxIm2,mVisu,false,mFinal);
           }
           else
           {
              mPIm2 = new cCEM_OneIm_Nuage(this,mNameIm2,mNameIm1,aBoxIm2,mVisu,false);
           }
       }
       else
       {
            mPIm2 = nullptr;
       }

       if (mPIm2)
       {
           mIm1->SetConj(mPIm2);
           mIm1->PostInit();
           mPIm2->PostInit();
       }

       if (HasHom)
       {
           ElPackHomologue aNewPack;
           Box2dr aBoxR1 = I2R(mBoxIm1);
           Box2dr aBoxR2 = I2R(aBoxIm2);

           for (ElPackHomologue::const_iterator itH = aPackH.begin() ;  itH!=aPackH.end(); itH++)
           {
               Pt2dr aP1 = itH->P1() / mDeZoom;
               Pt2dr aP2 = itH->P2() / mDeZoom;
               if (aBoxR1.inside(aP1) && aBoxR2.inside(aP2))
               {
                    aNewPack.Cple_Add(ElCplePtsHomologues(aP1-aBoxR1._p0,aP2-aBoxR2._p0));
               }
           }
           aPackH = aNewPack;
           mIm1->UsePack(aPackH);
       }


       Im2D_U_INT1 anAR1 = mIm1->ImAR();


       if (mFinal)
       {
          if (mCple)
          {
             Fonc_Num aFPx = mIm1->ImPx().in() * mIm1->ResolAlti();
             bool I1ISFirst = mCple->IsIm1(mNameIm1);
             Fonc_Num aFBsH = mCple->BSurHOfPx(I1ISFirst,aFPx);
             aFBsH = Max(0,Min(1,(aFBsH-mBSHRejet) / (mBSHOk-mBSHRejet))) * 255;
             ELISE_COPY(anAR1.all_pts(),Min(anAR1.in(),aFBsH),anAR1.out());
          }
          else
          {
               if (mNoOri)
               {
               }
               else
               {
                   ELISE_ASSERT(false,"Unachieved Coher Epip for non epipolar case");
               }
          }


          if (mRegulCheck)
          {
              Im2D_Bits<1>  aMasqL =  TestLabel(mIm1->ImPx_u2(),1<<16);
              ELISE_COPY(aMasqL.all_pts(),aMasqL.in()*anAR1.in(),anAR1.out());
          }
          else
          {
          }
       }


       Tiff_Im::Create8BFromFonc(mDir+ mPrefix + mPostfixP + ".tif",anAR1.sz(),anAR1.in());


       if (mDoMasq)
       {
           // double aMul = 20;
           // mReduce = 2.0;

           Pt2di aSz0 = anAR1.sz();
           Pt2di aSzR = round_up(Pt2dr(aSz0)/mReduceM);
           Im2D_REAL4  anArRed(aSzR.x,aSzR.y);

           Fonc_Num FScore = anAR1.in_proj();

           ELISE_COPY
           (
                anArRed.all_pts(),
                StdFoncChScale_Bilin(FScore,Pt2dr(0,0),Pt2dr(mReduceM,mReduceM)),
                anArRed.out()
           );

           Im2D_Bits<1> aMassFR(anAR1.sz().x,anAR1.sz().y);

           TIm2D<REAL4,REAL8>  aTArRed(anArRed);
           cOptimLabelBinaire * anOLB = cOptimLabelBinaire::CoxRoy(aSzR,0.5,mRegul);
           // cOptimLabelBinaire * anOLB = cOptimLabelBinaire::ProgDyn(aSzR,0.5,mRegul);

           Pt2di aP;

           for (aP.x=0 ;  aP.x<aSzR.x ; aP.x++)
               for (aP.y=0 ;  aP.y<aSzR.y ; aP.y++)
                     anOLB->SetCost(aP,aTArRed.get(aP));


           Im2D_Bits<1> aISol = anOLB->Sol();
           ELISE_COPY(anAR1.all_pts(),aISol.in_proj()[Virgule(FX/mReduceM,FY/mReduceM)],aMassFR.out());

           delete anOLB;

/*
           Im2D_INT2 aIZMin(aSzR.x,aSzR.y,0);
           Im2D_INT2 aIZMax(aSzR.x,aSzR.y,3);
           Im2D_INT2 aISol(aSzR.x,aSzR.y);
           cInterfaceCoxRoyAlgo * aCox = cInterfaceCoxRoyAlgo::NewOne
                                     (
                                         aSzR.x,
                                         aSzR.y,
                                         aIZMin.data(),
                                         aIZMax.data(),
                                         true,
                                         false
                                     );
           REAL4 ** aData = anArRed.data();
           for (int anX = 0 ;  anX < aSzR.x ; anX++)
           {
               for (int anY = 0 ;  anY < aSzR.y ; anY++)
               {
                  aCox->SetCostVert(anX,anY,0,round_ni(0.5*aMul));
                  double aCost = 1- aData[anY][anX] /255.0;
                  aCox->SetCostVert(anX,anY,1,round_ni(aCost*aMul));
                  aCox->SetCostVert(anX,anY,2,round_ni(aMul*2));
               }
           }
           aCox->SetStdCostRegul(0,aMul*mRegul,0);
           aCox->TopMaxFlowStd(aISol.data());

           ELISE_COPY(anAR1.all_pts(),aISol.in_proj()[Virgule(FX/mReduceM,FY/mReduceM)],aMassFR.out());
*/


           if (mFinal)
           {
              mImQualDepth = mIm1->VerifProf(aMassFR);
              Tiff_Im::Create8BFromFonc
              (
                  mDir+ mPrefix + "_ImDistor" + mPostfixP + ".tif",
                  anAR1.sz(), Min(255,100*mImQualDepth.in())
              );

              if (mFilterBH)
              {
                   Fonc_Num aF = MasqBorHomogene(mIm1->Im(),aMassFR,mIm1->Win());
                   if (mIm1->Win())
                   {
                      ELISE_COPY(select(aMassFR.all_pts(),aF),P8COL::yellow,mIm1->Win()->odisc());
                      getchar();
                   }
                   ELISE_COPY(select(aMassFR.all_pts(),aF),0,aMassFR.out());
               }
              // mIm1->VerifIm(aMassFR);
           }

           Tiff_Im::Create8BFromFonc
           (
               mDir+ mPrefix + "_Masq1" + mPostfixP + ".tif",
               anAR1.sz(), aMassFR.in()      // aISol.in_proj()[Virgule(FX/mReduce,FY/mReduce)]
           );

           // Creation du masque symetrique

           if (mDoMasqSym)
           {
                 ELISE_ASSERT(mPIm2!=nullptr,"Need adapation 4 MasqSym with empty data");
                 ELISE_COPY(aISol.all_pts(),cont_vect(aISol.in(),this,true),Output::onul());
                 std::sort(mConts.begin(),mConts.end());
                 Im2D_U_INT1 aMasq2(mPIm2->Sz().x,mPIm2->Sz().y,0);
                 for (int aK=0 ; aK<int(mConts.size()) ; aK++)
                 {
                     ELISE_COPY(polygone(*(mConts[aK].mL)),mConts[aK].mExt ? 1 : 0 , aMasq2.out());
                 }
                 Tiff_Im::Create8BFromFonc(mDir+mPrefix +"_Masq2"+mPostfixP+".tif", aMasq2.sz(),aMasq2.in());
                 cMTDCoher aMTD;
                 aMTD.Dec2() = aBoxIm2._p0;
                 std::string aNameXML = mDir+mPrefix + "_DEC2"+mPostfixP + ".xml";
                 MakeFileXML(aMTD,aNameXML);
           }
       }
   }

}

std::string cCoherEpi_main::NameIm(bool First)
{
   return First ? mNameIm1 : mNameIm2;
}

cInterfChantierNameManipulateur * cCoherEpi_main::ICNM() {return mICNM;}
cMasqBin3D * cCoherEpi_main::Masq3d() {return mMasq3d;}

const std::string & cCoherEpi_main::Ori() const
{
    return mOri;
}




void cCoherEpi_main::action(const  ElFifo<Pt2di> & aFil,bool ext)
{

    int aNbOk = 0;
    int aNbPasOk = 0;
    ElList<Pt2di> *  aLCont = new ElList<Pt2di>;
    ElFifo<Pt2dr> aVSurf;
    aVSurf.set_circ(true);

    for (int aK=0 ; aK<aFil.nb() ; aK++)
    {
        bool Ok;
        Pt2dr aP2 = mIm1->ToIm2(Pt2dr(aFil[aK])*mReduceM,Ok);
        if (Ok)
        {
           aNbOk++;
           *aLCont = *aLCont+Pt2di(aP2);
           aVSurf.push_back(aP2);
        }
        else
        {
           aNbPasOk++;
        }
    }
    cOneContour aC;
    aC.mSurf = ElAbs(surf_or_poly(aVSurf));
    aC.mExt  = ext;
    aC.mL    = aLCont;
    mConts.push_back(aC);
}


int CoherEpi_main(int argc,char ** argv)
{
    cCoherEpi_main aCEM(argc,argv);

    return 0;
}








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
