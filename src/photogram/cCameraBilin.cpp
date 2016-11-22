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

/*
   La distorsion d'une camera Bilineaire est codee par la valeur du deplacement sur les N Noeuds.

   Le modele bilineaire a ete prefere au modele triangule car :

      * plus simple a gerer au niveau des indexe
      * naturellement continu en extrapolation
      * ni pire ni meilleur  que la triangulation en continuite
      

   Analyse de phgr_dist_unif.h

         cDist_Param_Unif_Gen => ElDistortion22_Gen
         class cCamera_Param_Unif_Gen :  public CamStenope

class cPIF_Unif_Gen : public cParamIntrinsequeFormel



*/

static const Pt2di CamBilinCorn[4]={Pt2di(0,0),Pt2di(1,0),Pt2di(0,1),Pt2di(1,1)};

bool DebugCamBil = false;

/*

class cEqAffine
{
    public :
        cEqAffine(cSetEqFormelles * aSet,int aNbInc,bool GenCode);

    private  :
        std::vector<cVarEtat_PhgrF>   mEtaLin;
        cVarEtat_PhgrF                mEtatCste;
};
*/

/**************************************************************/
/*                                                            */
/*                 cPIF_Bilin      :                          */
/*                                                            */
/**************************************************************/


/*
cEqAffine::cEqAffine(cSetEqFormelles * aSet,int aNbInc,bool GenCode)
{
}
*/

/**************************************************************/
/*                                                            */
/*                 cPIF_Bilin,cQuadrangle                     */
/*                                                            */
/**************************************************************/
cSomBilin::cSomBilin(cSetEqFormelles & aSet,Pt2dr & aPt,const cIncIntervale & anInt)  :
   mPtF        (aSet.Alloc().NewPt2("cSomBilin",aPt)),
   mInterv     (anInt)
{
}


cQuadrangle::cQuadrangle
(
     const cIncIntervale & aI00,
     const cIncIntervale & aI10,
     const cIncIntervale & aI01,
     const cIncIntervale & aI11
) :
   mInt00  (aI00),
   mInt10  (aI10),
   mInt01  (aI01),
   mInt11  (aI11)
{
}


/**************************************************************/
/*                                                            */
/*                 cPIF_Bilin      :                          */
/*                                                            */
/**************************************************************/
extern bool AllowUnsortedVarIn_SetMappingCur;


static std::string NameInterv(const Pt2di & aP)
{
   return std::string("Bil"+ToString(aP.x)+"_"+ToString(aP.y));
}

cPIF_Bilin::cPIF_Bilin(cCamStenopeBilin *aCSB,cSetEqFormelles & aSet):
    cParamIntrinsequeFormel (false,aCSB,aSet,true),
    mSet                    (aSet),
    mFiged                  (false),
    mDistInit               (aCSB->DBL()),
    mDistCur                (aCSB->DBL()),
    mCurPIF                 (0),
    mFVDist                 (1+mDistCur.Nb().y),
    mQuads                  (mDistCur.Nb().y),
    mLastCase               (-1,-1)
{
    AllowUnsortedVarIn_SetMappingCur = true;
    SetFocFree(true);
    SetPPFree(true);
 
    for (int aK=0 ; aK<4 ; aK++)
    {
        mCornF.push_back(cP2d_Etat_PhgrF(std::string("Pts")+ToString(aK)));
    }

    int aCpt=0;
    // Pour initier dans unr ordre ou le premri quadrangle passe d'abord
    for (int aKD8=0 ; aKD8<=ElMax(mDistCur.Nb().y,mDistCur.Nb().x) ; aKD8++)
    {
        for (int aKY=0; aKY<=mDistCur.Nb().y ; aKY++)
        {
            for (int aKX=0; aKX<=mDistCur.Nb().x ; aKX++)
            {
                if (ElMax(aKX,aKY)==aKD8)
                {
                    if (aKY == (mDistCur.Nb().y/2))
                    {
                        if (aKX==0)
                           mIndFrozen0 = 2 * aCpt;
                        if (aKX==mDistCur.Nb().x)
                           mIndFrozen1 = 2 * aCpt;
                    }
                    cIncIntervale anInt(false,NameInterv(Pt2di(aKX,aKY)),aSet,2);
                // anInt.Close();
                    cSomBilin aSB(mSet,mDistCur.Dist(Pt2di(aKX,aKY)),anInt);
                // aSB.mPtF = mSet.Alloc().NewPt2(mDistCur.Dist(Pt2di(aKX,aKY)));
                    mFVDist[aKY].push_back(aSB);
                    aCpt++;
                // mFVDist.push_back(mSet.Alloc().NewPt2(mDistCur.Dist(Pt2di(aKX,aKY))));
                }
            }
        }
    }

    Pt2di aC;
    for (aC.y=0; aC.y<mDistCur.Nb().y ; aC.y++)
    {
        for (aC.x=0; aC.x<mDistCur.Nb().x ; aC.x++)
        {
             cIncIntervale  anI00(FDist(aC+CamBilinCorn[0]).mInterv,NameInterv(CamBilinCorn[0]));
             cIncIntervale  anI10(FDist(aC+CamBilinCorn[1]).mInterv,NameInterv(CamBilinCorn[1]));
             cIncIntervale  anI01(FDist(aC+CamBilinCorn[2]).mInterv,NameInterv(CamBilinCorn[2]));
             cIncIntervale  anI11(FDist(aC+CamBilinCorn[3]).mInterv,NameInterv(CamBilinCorn[3]));

             mQuads[aC.y].push_back(cQuadrangle(anI00,anI10,anI01,anI11));
        }
    }

    NV_UpdateCurPIF();
}


void cPIF_Bilin::PrepareEqFForPointIm(const cIncListInterv & aII0,cElCompiledFonc * aFonc,const Pt2dr & aPIm,bool EqDroite,int aKCam)
{
   //  std::cout << "PPprePP " << this << " " << aFonc << "\n";

    // std::cout << "JJJJJJjjjJJjKK " << aFonc->NameAlloc() << "\n";

    ELISE_ASSERT(!EqDroite,"cPIF_Bilin do not handle eq droite!!");
    ELISE_ASSERT(aKCam==0,"cPIF_Bilin do not handle KCam!=0!!");

    mDistCur.InitEtatFromCorner(mDistCur.ToCoordGrid(aPIm));
    Pt2di aCorner = mDistCur.mCurCorner;

    int aOffs = aKCam*4;
    for (int aK=0 ; aK<4 ; aK++)
    {
          mCornF[aK+aOffs].InitAdrSVP(*aFonc);
          Pt2di aPInt = aCorner+CamBilinCorn[aK];
          mCornF[aK+aOffs].SetEtatSVP(mDistCur.FromCoordGrid(Pt2dr(aPInt)));

// std::cout << "Pprrp " << aPInt << " =>>" << mDistCur.FromCoordGrid(Pt2dr(aPInt)) << "\n";
    }

    // ON NE PEUT PAS UTILISER mLastCase car aFonc est reutilise un peu partout !!!
    // if (aCorner== mLastCase) return;

    mLastCase = aCorner;


    cIncListInterv aNewI(aII0);
    const cQuadrangle  aQ= mQuads[aCorner.y][aCorner.x];
    aNewI.ResetInterv(aQ.mInt00);
    aNewI.ResetInterv(aQ.mInt10);
    aNewI.ResetInterv(aQ.mInt01);
    aNewI.ResetInterv(aQ.mInt11);
    aFonc->SetMappingCur(aNewI,&mSet);
    
     // std::cout << "PPprePP  " << aPIm << " " << aCorner << "\n";
}

cSomBilin & cPIF_Bilin::FDist(const Pt2di & aP)
{
   return mFVDist[aP.y][aP.x];
}

Pt2d<Fonc_Num> cPIF_Bilin::VirtualDist(Pt2d<Fonc_Num> aP,bool UsePC,int aKCam)
{
   ELISE_ASSERT(aKCam==0,"cPIF_Bilin do not handle KCam!=0!!");
   int aOffs = aKCam * 4;

   Pt2d<Fonc_Num> aP00 = mCornF[aOffs].PtF();
   Pt2d<Fonc_Num> aP10 = mCornF[aOffs+1].PtF();
   Pt2d<Fonc_Num> aP01 = mCornF[aOffs+2].PtF();

   Fonc_Num aPdsX = (aP10.x-aP.x) / (aP10.x-aP00.x);
   Fonc_Num aPdsY = (aP01.y-aP.y) / (aP01.y-aP00.y);

   return    FDist(Pt2di(0,0)).mPtF.mul(aPdsX     * aPdsY    )
          +  FDist(Pt2di(1,0)).mPtF.mul((1-aPdsX) * aPdsY    )
          +  FDist(Pt2di(0,1)).mPtF.mul(aPdsX     * (1-aPdsY))
          +  FDist(Pt2di(1,1)).mPtF.mul((1-aPdsX) * (1-aPdsY))  ;

/*
    return     mFVDist[0].mPtF.mul(mPds[0+aOffs].FN())
            +  mFVDist[1].mPtF.mul(mPds[1+aOffs].FN())
            +  mFVDist[2].mPtF.mul(mPds[2+aOffs].FN())
            +  mFVDist[3].mPtF.mul(mPds[3+aOffs].FN());
*/
}

void cPIF_Bilin::AddToListInterval(cIncListInterv & aLInterv)
{
   int aI0Alloc = mIncInterv.I0Alloc();
   cIncIntervale aI("Intr",aI0Alloc,aI0Alloc+3);
   
   aLInterv.AddInterv(aI);
   const cQuadrangle  aQ= mQuads[0][0];
   aLInterv.AddInterv(aQ.mInt00);
   aLInterv.AddInterv(aQ.mInt10);
   aLInterv.AddInterv(aQ.mInt01);
   aLInterv.AddInterv(aQ.mInt11);
  
}


cPIF_Bilin::~cPIF_Bilin() {}

/*
*/

CamStenope * cPIF_Bilin::CurPIF()
{
    return mCurPIF;
}

CamStenope * cPIF_Bilin::DupCurPIF()
{
    //  Respecte le protocole standard pour le cas ou on ajoute qq ch de + complexe dans NV_UpdateCurPIF
     cCamStenopeBilin * aCSB = mCurPIF;
     NV_UpdateCurPIF();
     ElSwap(aCSB,mCurPIF);
  
    return aCSB;
}

void   cPIF_Bilin::NV_UpdateCurPIF()
{
   mCurPIF = new  cCamStenopeBilin(CurFocale(),CurPP(),mDistCur);
   mCurPIF->HeritComplAndSz(*CamInit());
}

void   cPIF_Bilin::UpdateCurPIF()  {NV_UpdateCurPIF();}

bool  cPIF_Bilin::IsDistFiged() const
{
    return mFiged;
}


const std::string cPIF_Bilin::TheNameType  = "CamBilin";
std::string  cPIF_Bilin::NameType() const { return TheNameType; }

void cPIF_Bilin::SetDistFigee()
{
   mFiged = true;
}

void cPIF_Bilin::SetDistFree(int aDegre)
{
   mFiged = false;
   mDegreFree = aDegre;  // Pour l'instant inutile
}


cMultiContEQF  cPIF_Bilin::StdContraintes()
{
   cMultiContEQF aRes;
   if (mFiged)
   {
        AddFoncRappInit(aRes,0,NbInc(),cContrainteEQF::theContrStricte);
   }
   else
   {
        AddFoncRappInit(aRes,mIndFrozen0,mIndFrozen0+2,cContrainteEQF::theContrStricte);
        AddFoncRappInit(aRes,mIndFrozen1,mIndFrozen1+2,cContrainteEQF::theContrStricte);
   }

   return aRes;
}

cPIF_Bilin * cSetEqFormelles::NewPIFBilin(cCamStenopeBilin * aCSB)
{
   cPIF_Bilin * aRes = new cPIF_Bilin(aCSB,*this);
   aRes->CloseEEF();
   AddObj2Kill(aRes);


   return aRes;
}

cParamIntrinsequeFormel *  cSetEqFormelles::AsPIF_NewPIFBilin(cCamStenopeBilin * aCSB)
{
   return NewPIFBilin(aCSB);
}

cParamIntrinsequeFormel * GenCodePIFCamBilin(cSetEqFormelles * aSet)
{
     cDistorBilin aDBL(Pt2dr(3000,2000),Pt2dr(0,0),Pt2di(1,1));
     cCamStenopeBilin  * aCBL = new cCamStenopeBilin(5000,Pt2dr(1500,1000),aDBL);

     return aSet->NewPIFBilin(aCBL);
}


/**************************************************************/
/*                                                            */
/*                 cCamStenopeBilin:                          */
/*                                                            */
/**************************************************************/

std::vector<double> NoAF;

cCamStenopeBilin::cCamStenopeBilin
(
     REAL Focale,
     Pt2dr Centre,
     const  cDistorBilin & aDBL
) :
  CamStenope  (true,Focale,Centre,NoAF),
  mDBL        (aDBL)
{
}

const ElDistortion22_Gen & cCamStenopeBilin::Dist() const {return mDBL;}
ElDistortion22_Gen & cCamStenopeBilin::Dist()  {return mDBL;}
const cDistorBilin & cCamStenopeBilin::DBL() const  {return mDBL;}

cCamStenopeBilin * cCamStenopeBilin::CSBil_SVP()
{
    return this;
}

/**************************************************************/
/*                                                            */
/*                 cDistorBilin                               */
/*                                                            */
/**************************************************************/

cDistorBilin::cDistorBilin(Pt2dr aP0,Pt2dr aP1,Pt2di aNb) :
   mP0     (aP0),
   mP1     (aP1),
   mStep   ((aP1-aP0).dcbyc(Pt2dr(aNb))),
   mNb     (aNb)
{

    for (int aKY=0 ; aKY<= mNb.y ; aKY++)
    {
        for (int aKX=0 ; aKX<= mNb.x ; aKX++)
        {
            mVDist.push_back(FromCoordGrid(Pt2dr(aKX,aKY)));
        }
    }
}

Pt2dr cDistorBilin::ToCoordGrid(const Pt2dr & aP) const   { return (aP-mP0).dcbyc(mStep); } 
Pt2dr cDistorBilin::FromCoordGrid(const Pt2dr & aP) const { return  mP0+aP.mcbyc(mStep); } 


void  cDistorBilin::GetDebInterval(int & aX0,const int & aSzGrd,const double & aCoordGr) const
{
   aX0 =  ElMax(0,ElMin(aSzGrd-1,round_down(aCoordGr)));
}


void cDistorBilin::GetDebIntervalAndPds(int & aX0,double & aPdsX0,const int & aSzGrd,const double & aCoordGr) const
{
    GetDebInterval(aX0,aSzGrd,aCoordGr);
    aPdsX0 = 1.0 - (aCoordGr-aX0);
}

void  cDistorBilin::GetParamCorner(Pt2di & aCornerBG,Pt2dr & aPdsBG,const Pt2dr & aCoorGr) const
{
     GetDebIntervalAndPds(aCornerBG.x,aPdsBG.x,mNb.x,aCoorGr.x);
     GetDebIntervalAndPds(aCornerBG.y,aPdsBG.y,mNb.y,aCoorGr.y);
}

void cDistorBilin::InitEtatFromCorner(const Pt2dr & aCoorGr) const
{
   Pt2dr aPds;
   GetParamCorner(mCurCorner,aPds,aCoorGr);
   mPds[0] = aPds.x * aPds.y;
   mPds[1] = (1-aPds.x) * aPds.y;
   mPds[2] = aPds.x * (1-aPds.y);
   mPds[3] = (1-aPds.x) * (1-aPds.y);
    
}
Pt2dr cDistorBilin::Direct(Pt2dr aP) const
{
    InitEtatFromCorner(ToCoordGrid(aP));

    return   
             Dist(mCurCorner +CamBilinCorn[0] ) * mPds[0]
           + Dist(mCurCorner +CamBilinCorn[1] ) * mPds[1]
           + Dist(mCurCorner +CamBilinCorn[2] ) * mPds[2]
           + Dist(mCurCorner +CamBilinCorn[3] ) * mPds[3];
}


void  cDistorBilin::Diff(ElMatrix<REAL> & aM,Pt2dr aP) const
{
    // InitEtatFromCorner(ToCoordGrid(aP));

    Pt2dr aPds;
    GetParamCorner(mCurCorner,aPds,ToCoordGrid(aP)); 
    const Pt2dr & aP00 =  Dist(mCurCorner+CamBilinCorn[0] ) ;
    const Pt2dr & aP10 =  Dist(mCurCorner+CamBilinCorn[1] ) ;
    const Pt2dr & aP01 =  Dist(mCurCorner+CamBilinCorn[2] ) ;
    const Pt2dr & aP11 =  Dist(mCurCorner+CamBilinCorn[3] ) ;


    Pt2dr aGx =    ((aP10-aP00)*aPds.y + (aP11-aP01)*(1-aPds.y))  / mStep.x;
    Pt2dr aGy =    ((aP01-aP00)*aPds.x + (aP11-aP10)*(1-aPds.x))  / mStep.y;

    aM.ResizeInside(2,2);
    SetCol(aM,0,aGx);
    SetCol(aM,1,aGy);

    // A conserver, verification par diff std
    if (0)
    {
        ElMatrix<REAL> aM2(2,2);
        DiffByDiffFinies(aM2,aP,euclid(mStep)/1e4);
        static double aDMax = 0;
        double aD = aM.L2(aM2);
        if (aD>aDMax)
        {
            aDMax = aD;
            std::cout << "DDDD " << aD << "\n";
        }
    }
    // InitEtatFromCorner(ToCoordGrid(aP));

}

void cDistorBilin::InitAffine(double aF,Pt2dr aPP)
{
   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
           Pt2di aPGrI(aKX,aKY);
           Pt2dr aPGrR(aPGrI);
           Pt2dr aPR = FromCoordGrid(aPGrR);

           Dist(aPGrI) = aPP + aPR*aF;
       }
   }
}

Box2dr cDistorBilin::BoxRab(double aMulStep) const
{
    Pt2dr aRab= mStep * aMulStep;
    return Box2dr (mP0-aRab,mP1+aRab);
}
  
void cDistorBilin::Randomize(double aFact)
{
   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
             Dist(Pt2di(aKX,aKY)) = FromCoordGrid(Pt2dr(aKX,aKY) + Pt2dr(NRrandC(),NRrandC()) * aFact);
       }
   }
}

extern cCalibDistortion GlobXmlDistNoVal();

cDistorBilin cDistorBilin::FromXmlGridStuct(const cCalibrationInterneGridDef &  aCIG)
{
    cDistorBilin aCBL(aCIG.P0(),aCIG.P1(),aCIG.Nb());

    int aNbPtGr = (int)aCIG.PGr().size();
    Pt2di aNb = aCIG.Nb();

    if (aNbPtGr==((aNb.y+1)*(aNb.x+1)))
    {
        int aCpt=0;
        for (int aKY=0 ; aKY<= aNb.y ; aKY++)
        {
            for (int aKX=0 ; aKX<= aNb.x ; aKX++)
            {
                  aCBL.Dist(Pt2di(aKX,aKY)) = aCIG.PGr()[aCpt++];
            }
        }
    }
    else if (aNbPtGr==0)
    {
    }
    else
    {
        std::cout << "Sz=" << aNbPtGr << "\n";
        ELISE_ASSERT(false,"Bad size for cDistorBilin::FromXmlGridStuct");
    }

    return aCBL;
}

CamStenope * GlobFromXmlGridStuct(REAL aFoc,Pt2dr aCentre,const cCalibrationInterneGridDef &  aCIG)
{
    return new cCamStenopeBilin(aFoc,aCentre,cDistorBilin::FromXmlGridStuct(aCIG));
}


cCalibrationInterneGridDef  cDistorBilin::ToXmlGridStruct() const
{
   cCalibrationInterneGridDef aRes;
   aRes.P0() = mP0;
   aRes.P1() = mP1;
   aRes.Nb() = mNb;

   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
           aRes.PGr().push_back(Dist(Pt2di(aKX,aKY)));
       }
   }

   return aRes;
}


cCalibDistortion FromCIGD(const cCalibrationInterneGridDef & aCIGD)
{
    cCalibDistortion  aRes = GlobXmlDistNoVal();
    aRes.ModGridDef().SetVal(aCIGD);

    return aRes;
}


cCalibDistortion cDistorBilin::ToXmlStruct(const ElCamera *) const
{
   return FromCIGD(ToXmlGridStruct());
}

bool  cDistorBilin::AcceptScaling() const {return true; }
bool  cDistorBilin::AcceptTranslate() const {return true; }

/*

   Extrait de photogram.h :
     Soit H (X) == PP + X * F   se transforme en H-1 D H

     Pt2dr cDistorBilin::ToCoordGrid(const Pt2dr & aP) const   { return (aP-mP0).dcbyc(mStep); } 
    ( PP + X * F -mP0) / S  = (X-P')/S'
    (PP -mP0)/S = -P' *F/S

    P' = (mP0 -PP) /F   ; S' = S/F     

*/

void cDistorBilin::V_SetScalingTranslate(const double & aF,const Pt2dr & aPP)
{
   for (int aKY=0 ; aKY<= mNb.y ; aKY++)
   {
       for (int aKX=0 ; aKX<= mNb.x ; aKX++)
       {
           Dist(Pt2di(aKX,aKY)) = ( Dist(Pt2di(aKX,aKY))- aPP) / aF;
       }
   }
   mP0 = (mP0-aPP) / aF;
   mP1 = (mP1-aPP) / aF;
   mStep = mStep / aF;
}

void Test_DBL()
{
    Pt2dr aP0(-10,-20);
    Pt2dr aP1(1500,2000);
    Pt2di aNb(10,15);

    cDistorBilin aDBL1(aP0,aP1,aNb);
    Box2dr aBoxRab1 = aDBL1.BoxRab(0.3);

   //======================================================
   // Verif interpol/extrapol de fon lineaire est exacte 
   //======================================================

    for (int aTime=0 ; aTime<10000 ; aTime++)
    {
        double aF = pow(2.0,NRrandC()*8);
        Pt2dr aPP = Pt2dr(NRrandC(),NRrandC()) * aF;
        aDBL1.InitAffine(aF,aPP);
        for (int aK=0 ; aK<10; aK++)
        {
            Pt2dr aP0 = aBoxRab1.RandomlyGenereInside();
            Pt2dr aP1 = aDBL1.Direct(aP0);
            Pt2dr aQ1 = aPP + aP0 * aF;
            double aDist = euclid(aP1,aQ1);
            if (aDist>1e-9)
            {
                ELISE_ASSERT(false,"Test_DBL Affine");
            }
        }
    }




   //============================
   // Test copy
   //============================

    for (int aK=0 ; aK<10000 ; aK++)
    {
         aDBL1.Randomize();
         cDistorBilin aDBL2 = aDBL1;
         Pt2dr aP0 = aBoxRab1.RandomlyGenereInside();
         Pt2dr aP1 = aDBL1.Direct(aP0);
         Pt2dr aP2 = aDBL2.Direct(aP0);
         double aDist = euclid(aP1,aP2);
         ELISE_ASSERT(aDist==0,"Test_DBL dist");
    }
    
   //============================
   //  V_SetScalingTranslate
   //============================

    for (int aTime=0 ; aTime<10000 ; aTime++)
    {
        double aF = pow(2.0,NRrandC()*8);
        Pt2dr aPP = Pt2dr(NRrandC(),NRrandC()) * aF;
        aDBL1.Randomize();
        cDistorBilin aDBL2 = aDBL1;
        aDBL2.V_SetScalingTranslate(aF,aPP);
        Box2dr aBoxRab2 = aDBL2.BoxRab(0.3);

        for (int aK=0 ; aK<10; aK++)
        {
            Pt2dr aP0 = aBoxRab2.RandomlyGenereInside();
            Pt2dr aP2 = aDBL2.Direct(aP0);

            Pt2dr aP1 = (aDBL1.Direct(aPP+aP0*aF)-aPP) /aF;
            double aDist = euclid(aP1 - aP2);

            ELISE_ASSERT(aDist<1e-9,"DBL-setScalingTranslate");
        }
    }

   //============================
   //  Verif Inverse
   //============================

    for (int aTime=0 ; aTime<100000 ; aTime++)
    {
        aDBL1.Randomize(0.01);

        for (int aK=0 ; aK<10; aK++)
        {
            Pt2dr aP0 = aBoxRab1.RandomlyGenereInside();
            Pt2dr aP1 = aDBL1.Direct(aP0);
            Pt2dr aP2 = aDBL1.Inverse(aP1);

            double aDist = euclid(aP0 - aP2);
            // std::cout << "D= " << aDist << "\n";

            ELISE_ASSERT(aDist<1e-5,"DBL-setScalingTranslate");
        }
   }
/*
*/


    std::cout << "DONE Test cDistorBilin\n";
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
