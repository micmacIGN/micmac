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
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/


#include "StdAfx.h"
#include "hassan/reechantillonnage.h"

static const double MulCost = 1e1;

class cElemFDBPD
{
     public :
        cElemFDBPD(bool isInside) :
            mInside  (isInside)
        {
        }
        cElemFDBPD() :
            mInside  (false)
        {
        }

        void InitTmp(const cTplCelNapPrgDyn<cElemFDBPD> & aCel)
        {
            *this = aCel.ArgAux();
        }


        U_INT1  mInside;
    
};

class cFiltrageDepthByProgDyn
{
     public :
        typedef  cElemFDBPD tArgCelTmp;
        typedef  cElemFDBPD tArgNappe;
        typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
        typedef  cTplCelOptProgDyn<tArgCelTmp>  tCelOpt;


        // aImLab => 0 Out, 1 In, 2 Interpole (donc + favorable a devenir un trou)

        cFiltrageDepthByProgDyn(Im2D_REAL4 aImDepth,Im2D_U_INT1 aImLab,const cParamFiltreDepthByPrgDyn & aParam);

        void DoConnexion
           (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*Input,int aInZMin,int aInZMax,
                  tCelOpt*Ouput,int aOutZMin,int aOutZMax
           );
       void GlobInitDir(cProg2DOptimiser<cFiltrageDepthByProgDyn> &) {}
       Im2D_Bits<1> Result() const {return mResult;}
     private : 


        int ToICost(double aCost) {return round_ni(MulCost * aCost);}
        double ToRCost(int    aCost) {return aCost/ MulCost;}


        Pt2di              mSz;
        Im2D_U_INT1        mImLabel;  // 0 => Rien, 1= > OK, 2=> trou bouche
        TIm2D<U_INT1,INT>  mTLab;
        Im2D_REAL4         mImDepth;   // image dequant et trous bouches
        TIm2D<REAL4,REAL>  mTDetph;
        // Cout de reference 1 pour etre non affecte, 0 pour etre conserve
        double             mParamCostNonAf;
        double             mParamCostTrans;
        double             mParamCostRegul;
        double             mPasDZ;  // Pour simplifier le parametrage compte tenu du fait que la dynamique est tres variable
        double             mMaxDZ;  // Au dela d'une certain valeur ajoute pas en irreg
        int                mNbDir;
        Im2D_Bits<1>       mResult;
};

void cFiltrageDepthByProgDyn::DoConnexion
     (
                  const Pt2di & aPIn, const Pt2di & aPOut,
                  ePrgSens aSens,int aRab,int aMul,
                  tCelOpt*aTabInput,int aInZMin,int aInZMax,
                  tCelOpt*aTabOuput,int aOutZMin,int aOutZMax
     )
{
    int aLabIn = mTLab.get(aPIn);
    int aLabOut = mTLab.get(aPOut);

    bool WithUndef = (aLabIn==0) || (aLabOut==0) ; // Aucune communication avec le reste des que un outside

    for (int aZIn=0 ; aZIn<aInZMax ; aZIn++)
    {
        tCelOpt & anInp = aTabInput[aZIn];
        const  tArgCelTmp & aCelIn = anInp.ArgAux();
        for (int aZOut=0 ; aZOut<aOutZMax ; aZOut++)
        {
            tCelOpt & anOut = aTabOuput[aZOut];
            const  tArgCelTmp & aCelOut = anOut.ArgAux();
            double aCost = 0.0;
            if (!WithUndef)
            {
                 if (aCelIn.mInside != aCelOut.mInside)
                    aCost = mParamCostTrans;
                 else if (aCelIn.mInside  && aCelOut.mInside)
                 {
                    double aDZ = ElAbs(mTDetph.get(aPIn)-mTDetph.get(aPOut)) * mPasDZ ;
                    // Fonction concave, derivee nulle en zero,  assympote en mMaxDZ
                    // aDZ =  (sqrt(1+aDZ/mMaxDZ)-1) * 2*mMaxDZ ;
                    aDZ =  (mMaxDZ * aDZ) / (mMaxDZ + aDZ);
                    aCost =  mParamCostRegul * aDZ;
                 }
            }
            anOut.UpdateCostOneArc(anInp,aSens,ToICost(aCost));
/*
            double aDZ = ElAbs(aPIn.Z()-aPOut.Z())/mResolPlaniEquiAlt;
            if ((mFNoVal==0) || (aDZ < mFNoVal->PenteMax()))
            {
            // Fonction concave, nulle et de derivee 1 en 0
                 double aCost = (sqrt(1+aDZ/aSig0)-1) * 2*aSig0 * mFPrgD->Regul();
                 anOut.UpdateCostOneArc(anInp,aSens,ToICost(aCost));
            }
*/
        }
    }

}



cFiltrageDepthByProgDyn::cFiltrageDepthByProgDyn(Im2D_REAL4 aImDepth,Im2D_U_INT1 aImLab,const cParamFiltreDepthByPrgDyn & aParam) :
    mSz               (aImLab.sz()),
    mImLabel          (aImLab),
    mTLab             (mImLabel),
    mImDepth          (aImDepth),
    mTDetph           (mImDepth),
    mParamCostNonAf   (aParam.CostNonAff().Val()),
    mParamCostTrans   (aParam.CostTrans().Val()),
    mParamCostRegul   (aParam.CostRegul().Val()),
    mPasDZ            (aParam.StepZ()),
    mMaxDZ            (aParam.DzMax().Val()),
    mNbDir            (aParam.NbDir().Val()),
    mResult           (mSz.x,mSz.y)
{
// std::cout << "PARM COST REG " << mParamCostRegul << "\n";
   Im2D_INT2 aImMin(mSz.x,mSz.y,0);
   Im2D_INT2 aImMax(mSz.x,mSz.y);  
   ELISE_COPY(aImMax.all_pts(),1+(mImLabel.in()!=0),aImMax.out());

   cProg2DOptimiser<cFiltrageDepthByProgDyn>  * aPrgD = new cProg2DOptimiser<cFiltrageDepthByProgDyn>(*this,aImMin,aImMax,0,1); // 0,1 =>

   cDynTplNappe3D<cTplCelNapPrgDyn<cElemFDBPD> > & aNap = aPrgD->Nappe();


   Pt2di aP;
   for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
   {
       for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
       {
           int aLab = mTLab.get(aP);
           cTplCelNapPrgDyn<cElemFDBPD> * aTabP = aNap.Data()[aP.y][aP.x];

           if (aLab==0)
           {
                aTabP[0].ArgAux() = cElemFDBPD(false);
                aTabP[0].SetOwnCost(ToICost(0));
           }
           else
           {
                // Cout 1.0 arbitraire  pour etre refute
                aTabP[0].ArgAux() = cElemFDBPD(false);
                aTabP[0].SetOwnCost(ToICost( (aLab==2) ? mParamCostNonAf : 1.0));

                // Cout 0 pour etre garde
                aTabP[1].ArgAux() = cElemFDBPD(true);
                aTabP[1].SetOwnCost(ToICost(0));
           }
         
       }
   }
   aPrgD->DoOptim(mNbDir);
   Im2D_INT2 aSol(mSz.x,mSz.y);
   aPrgD->TranfereSol(aSol.data());

   ELISE_COPY(mResult.all_pts(),aSol.in() && (mImLabel.in()!=0) ,mResult.out());
}

Im2D_Bits<1>    FiltrageDepthByProgDyn(Im2D_REAL4 aImDepth,Im2D_U_INT1 aImLab,const cParamFiltreDepthByPrgDyn & aParam)
{
     cFiltrageDepthByProgDyn aFilter(aImDepth,aImLab,aParam);

     return aFilter.Result();
}


/***********************************************************************************/
/*                                                                                 */
/*                      FiltreDetecRegulProf                                       */
/*                                                                                 */
/***********************************************************************************/

class cCC_NbMaxIter : public  cCC_NoActionOnNewPt
{
   public  :
       cCC_NbMaxIter(int aNbMax) :
          mNbIter (0),
          mNbMaxIter (aNbMax)
       {
       }


       void OnNewStep() { mNbIter++;}
       void  OnNewPt(const Pt2di & aP) 
       {
           mVPts.push_back(aP);
       }
       bool  StopCondStep() {return mNbIter>=mNbMaxIter;}

    
       std::vector<Pt2di> mVPts;
       int                mNbIter;
       int                mNbMaxIter;
};


class cCCMaxAndBox : public  cCC_NbMaxIter
{
   public  :
       cCCMaxAndBox(int aNbMax,const Box2di & aBox) :
           cCC_NbMaxIter(aNbMax),
           mBox(aBox)
       {
       }
       bool ValidePt(const Pt2di & aP){return mBox.inside(aP);}

       Box2di mBox;
};


template <class tNum,class tNBase>  Im2D_Bits<1> TplFiltreDetecRegulProf
                                        (
                                             TIm2D<tNum,tNBase> aTProf, 
                                             TIm2DBits<1>  aTMasq,
                                             const cParamFiltreDetecRegulProf & aParam
                                        )
{
    FiltrageCardCC(true,aTMasq,1,0, aParam.NbCCInit().Val());

    Pt2di aSz = aTProf.sz();
    Im2D_Bits<1> aIMasq = aTMasq._the_im;

    Im2D_Bits<1> aMasqTmp = ImMarqueurCC(aSz);
    TIm2DBits<1> aTMasqTmp(aMasqTmp);
    bool V4= aParam.V4().Val();

    Im2D_REAL4 aImDif(aSz.x,aSz.y);
    TIm2D<REAL4,REAL8> aTDif(aImDif);

    ELISE_COPY(aIMasq.border(1),0,aIMasq.out());

    Pt2di aP;
    int aSzCC = aParam.SzCC().Val();
    double aPondZ = aParam.PondZ().Val();
    double aPente = aParam.Pente().Val();
    for (aP.x =0 ; aP.x < aSz.x ; aP.x++)
    {
        for (aP.y =0 ; aP.y < aSz.y ; aP.y++)
        {
             if (aTMasq.get(aP))
             {
                 cCC_NbMaxIter aCCParam(aSzCC);
                 OneZC(aP,V4,aTMasqTmp,1,0,aTMasq,1,aCCParam);
                 tNBase aZ0 =  aTProf.get(aP);
                 double aSomP = 0;
                 int    aNbP  = 0;
                 for (int aKP=0 ; aKP<int(aCCParam.mVPts.size()) ; aKP++)
                 {
                     const Pt2di & aQ = aCCParam.mVPts[aKP];
                     aTMasqTmp.oset(aQ,1);
                     if (aKP>0)
                     {
                         double aDist = euclid(aP,aQ);
                         double aDZ = ElAbs(aZ0-aTProf.get(aQ));
                         double aAttZ = aPondZ + aPente * aDist;
                         double aPds  = 1 / (1 + ElSquare(aDZ/aAttZ));
                         aNbP++;
                         aSomP += aPds;
                     }
                 }
                 aNbP = ElMax(aNbP,aSzCC*(1+aSzCC));
                 aTDif.oset(aP,aSomP/aNbP);
             }
        }
    }
    if (aParam.NameTest().IsInit())
    {
       Tiff_Im::Create8BFromFonc(aParam.NameTest().Val(),aSz,aImDif.in()*255);
    }

    Im2D_Bits<1> aIResult(aSz.x,aSz.y);
    ELISE_COPY(aIResult.all_pts(),(aImDif.in()> aParam.SeuilReg().Val()) && (aIMasq.in()) , aIResult.out());
    return aIResult;

}

Im2D_Bits<1>  FiltreDetecRegulProf(Im2D_REAL4 aImProf,Im2D_Bits<1> aIMasq,const cParamFiltreDetecRegulProf & aParam)
{
   return TplFiltreDetecRegulProf(TIm2D<REAL4,REAL8>(aImProf),TIm2DBits<1>(aIMasq),aParam);
}


/*
void TestFiltreRegul()
{
   Pt2di aP0(2000,500);
   Pt2di aSz(500,500);

   Video_Win * aW = 0;


   Tiff_Im aFileProf ("XXXXXFusion_NuageImProf_LeChantier_Etape_1.tif");
   Tiff_Im aFileMasq ("XXXXXFusion_NuageImProf_LeChantier_Etape_1_Masq.tif");

   Im2D_REAL4    aImProf(aSz.x,aSz.y);
   Im2D_Bits<1>  aMasq(aSz.x,aSz.y);


   ELISE_COPY(aImProf.all_pts(),trans(aFileProf.in(0),aP0),aImProf.out());
   ELISE_COPY(aMasq.all_pts(),trans(aFileMasq.in(0),aP0),aMasq.out());

   if (aW)
   {
       ELISE_COPY(aImProf.all_pts(),aImProf.in()*5,aW->ocirc());
       ELISE_COPY(select(aMasq.all_pts(),!aMasq.in()),P8COL::black,aW->odisc());
   }

   cParamFiltreDetecRegulProf aParam;
   //TplFiltreDetecRegulProf(TIm2D<REAL4,REAL8>(aImProf),TIm2DBits<1>(aMasq),aParam);
std::cout << "AAAaaaA\n";
   FiltreDetecRegulProf(aImProf,aMasq,aParam);
std::cout << "BBBbBb\n";
getchar();
}
*/

/***********************************************************************************/
/*                                                                                 */
/*                      ReduceImageProf                                            */
/*                                                                                 */
/***********************************************************************************/
//static double aDistMax=0;

template <class tNum,class tNBase>  Im2D_REAL4   TplFReduceImageProf
                                        (
                                             double aDifStd ,
                                             TIm2DBits<1>  aTMasq,
                                             TIm2D<tNum,tNBase> aTProf, 
                                             const Box2dr &aBox,
                                             double aScale,
                                             Im2D_REAL4    aImPds,
                                             std::vector<Im2DGen*>  aVNew,
                                             std::vector<Im2DGen*> aVOld
                                        )
{
    // double aDifStd = 0.5;
    // std::cout << "TO CHANGE DIFF STDD  " << aDifStd << "\n";

    TIm2D<REAL4,REAL8> aTPds(aImPds);
    Pt2di aSzOut = aImPds.sz();
    Im2D<tNum,tNBase> aIProf = aTProf._the_im;
    Pt2di aSzIn = aTProf.sz();
    Im2D_REAL4 aRes(aSzOut.x,aSzOut.y,0.0);
    TIm2D<REAL4,REAL8> aTRes(aRes);

    Im2D_Bits<1> aMasqTmpCC = ImMarqueurCC(aSzIn);
    TIm2DBits<1> aTMasqTmpCC(aMasqTmpCC);

    aVNew.push_back(&aRes);
    aVOld.push_back(&aIProf);

    int aSzCC = ElMax(3,round_up(aScale*2+1));

    Pt2di aPOut;
    for (aPOut.x = 0 ; aPOut.x<aSzOut.x ; aPOut.x++)
    {
        for (aPOut.y = 0 ; aPOut.y<aSzOut.y ; aPOut.y++)
        {
              aTPds.oset(aPOut,0.0);
              Pt2dr aPRIn (aPOut.x*aScale +aBox._p0.x,aPOut.y*aScale+aBox._p0.y);
              int aXInCentreI = ElMax(1,ElMin(aSzIn.x-2,round_ni(aPRIn.x)));
              int aYInCentreI = ElMax(1,ElMin(aSzIn.y-2,round_ni(aPRIn.y)));
              Pt2di aPII(aXInCentreI,aYInCentreI);

              int aXI0 = ElMax(0,aXInCentreI-aSzCC);
              int aYI0 = ElMax(0,aYInCentreI-aSzCC);
              int aXI1 = ElMin(aSzIn.x-2,aXInCentreI+aSzCC);
              int aYI1 = ElMin(aSzIn.y-2,aYInCentreI+aSzCC);

              // 1 calcul du barrycentre
              Pt2dr aBar(0,0);
              double aSomP=0;
              Pt2di aPIn;
              for (aPIn.x=aXI0 ; aPIn.x<=aXI1 ; aPIn.x++)
              {
                  for (aPIn.y=aYI0 ; aPIn.y<=aYI1 ; aPIn.y++)
                  {
                       if (aTMasq.get(aPIn))
                       {
                            aSomP++;
                            aBar = aBar + Pt2dr(aPIn);
                       }
                  }
              }
              Pt2di aNearest = aPII;
              if ((aSomP>=ElSquare(aScale)) && (aTMasq.get(aNearest)))
              {
                  cCCMaxAndBox  aCCParam(aSzCC,Box2di(Pt2di(aXI0,aYI0),Pt2di(aXI1,aYI1)));
                  OneZC(aNearest,true,aTMasqTmpCC,1,0,aTMasq,1,aCCParam);

                  std::vector<Pt2di> aVP = aCCParam.mVPts;
                  int aNbP = (int)aVP.size();

                  for (int aKP=0 ; aKP<aNbP ; aKP++)
                  {
                     aTMasqTmpCC.oset(aVP[aKP],1);
                  }

                  double aProfRef = aTProf.get(aNearest);


                  double aSomPds = 0;
                  std::vector<double> aVPds;
                  for (int aKP=0 ; aKP< aNbP ; aKP++)
                  {
                      const Pt2di & aP = aVP[aKP];
                      double aDist= euclid(aP-aNearest);
                      double aProf =  aTProf.get(aP);
                      double aDifNorm = ElAbs(aProf-aProfRef) /(aDifStd * (1+aDist/2.0));
                      double aPdsProf = 1.0;

                      if (aDifNorm<1)
                      {
                      }
                      else if (aDifNorm<3)
                      {
                         aPdsProf = (3-aDifNorm) /2.0;
                      }
                      else
                      {
                         aPdsProf = 0;
                      }
                  
                      double aDistNorm= ElMin(aDist/aScale,2.0);
                      double aPdsDist = (1+cos(aDistNorm * (PI/2.0)));

                      double aPds = aPdsDist * aPdsProf;
                      aSomPds += aPds;
                      aVPds.push_back(aPds);
                  }
                  aTPds.oset(aPOut,1.0);

                  for (int aKI=0 ; aKI <int(aVNew.size()) ; aKI++)
                  {
                      double aSomIP = 0;
                      Im2DGen * aIOld = aVOld[aKI];
                      for (int aKP=0 ; aKP< aNbP ; aKP++)
                            aSomIP += aVPds[aKP] * aIOld->GetR(aVP[aKP]);
                      aVNew[aKI]->SetR(aPOut,aSomIP/aSomPds);
                  }
              }
        }
    }

   
    return aRes;
}


Im2D_REAL4 ReduceImageProf(double aDifStd,Im2D_Bits<1> aIMasq,Im2D_REAL4 aImProf, const Box2dr &aBox,double aScale,Im2D_REAL4 aImPds,std::vector<Im2DGen*>  aVNew,std::vector<Im2DGen*> aVOld)

{
   return TplFReduceImageProf(aDifStd,TIm2DBits<1>(aIMasq),TIm2D<REAL4,REAL8>(aImProf),aBox,aScale,aImPds,aVNew,aVOld);
}



Im2D_REAL4 ReduceImageProf(double aDifStd,Im2D_Bits<1> aIMasq,Im2D_INT2 aImProf, const Box2dr &aBox,double aScale,Im2D_REAL4 aImPds,std::vector<Im2DGen*>  aVNew,std::vector<Im2DGen*> aVOld)

{
   return TplFReduceImageProf(aDifStd,TIm2DBits<1>(aIMasq),TIm2D<INT2,INT>(aImProf),aBox,aScale,aImPds,aVNew,aVOld);
}




template <class tNum,class tNBase>  Im2D_REAL4   TplProlongByCont
                                           (
                                                Im2D_Bits<1>  &aMasqResul,
                                                Im2D_Bits<1> aIMasq,
                                                Im2D<tNum,tNBase> aInput,
                                                INT aNbProl,
                                                double aDistAdd,
                                                double aDistMaxAdd
                                            )
{
    Pt2di aSz = aIMasq.sz();
    ELISE_ASSERT(aSz==aInput.sz(),"Incohe sz in ProlongByCont");


    TIm2D<REAL4,REAL8> aTRes(aSz);
    Im2D_Bits<2> aMasqCalc(aSz.x,aSz.y);

    ELISE_COPY(aIMasq.all_pts(),aIMasq.in(),aMasqCalc.out());
    ELISE_COPY(aMasqCalc.border(1),2,aMasqCalc.out());
    ELISE_COPY(aInput.all_pts(),aInput.in(),aTRes._the_im.out());

    TIm2DBits<2> aTMC(aMasqCalc);

    // Recupere la frontiere
    std::vector<Pt2di>  aVCur;

    {
       Pt2di aP;
       for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
       {
           for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
           {
               if (aTMC.get(aP) == 0)
               {
                    bool Got = false;
                    for (int aKV=0 ; aKV<4 ; aKV++)
                    {
                        if (aTMC.get(aP+TAB_4_NEIGH[aKV]) == 1)
                           Got = true;
                    }
                    if (Got)
                    {
                       aVCur.push_back(aP);
                    }
               }
           }
       }
    }

    double aDCum = 0;
    while ((aNbProl>0)  && (!aVCur.empty()))
    {
         // Calcul la moyenne
         for (int aTime=0 ; aTime<5 ; aTime++)
         {
             std::vector<float> aVNewVals;
             for (int aKP=0 ; aKP<int(aVCur.size()) ; aKP++)
             {
                 double aSom = 0;
                 int    aNbV = 0;
                 Pt2di aP = aVCur[aKP];
                 for (int aKV=0 ; aKV<9 ; aKV++)
                 {
                     Pt2di aVois = aP+TAB_9_NEIGH[aKV];
                     if (aTMC.get(aVois)==1)
                     {
                        int aPds = PdsGaussl9NEIGH[aKV];
                        aSom +=  aTRes.get(aVois) * aPds;
                        aNbV += aPds;
                     }
                 }
                 ELISE_ASSERT(aNbV!=0,"Incoh vois in ProlongByCont");
                 aVNewVals.push_back(aSom/aNbV);
// if (Bug) std::cout << "P0  " << aP << " : " <<  aSom/aNbV << "\n\n";
             }
             for (int aKP=0 ; aKP<int(aVCur.size()) ; aKP++)
             {
                 Pt2di aP = aVCur[aKP];
                 aTRes.oset(aP,aVNewVals[aKP]+aDistAdd);
                 aTMC.oset(aP,1);  // apres la 1ere itere les dernier point sont inclus
             }
         }

         // Prochaine frontiere
         std::vector<Pt2di> aVNewGen;
         for (int aKP=0 ; aKP<int(aVCur.size()) ; aKP++)
         {
              for (int aKV=0 ; aKV<4 ; aKV++)
              {
                  Pt2di aVois = aVCur[aKP] + TAB_4_NEIGH[aKV];
                  if (aTMC.get(aVois) == 0)
                  {
                     aVNewGen.push_back(aVois);
                     aTMC.oset(aVois,1);
                  }
              }
         }
         for (int aKP=0 ; aKP<int(aVNewGen.size()) ; aKP++)
             aTMC.oset(aVNewGen[aKP],0);
         aNbProl--;
         aVCur = aVNewGen;
         aDCum += ElAbs(aDistAdd);
         if (aDCum> aDistMaxAdd) aDistAdd = 0;
    }

    aMasqResul =  Im2D_Bits<1>(aSz.x,aSz.y);
    ELISE_COPY(aMasqResul.all_pts(),aMasqCalc.in()==1,aMasqResul.out());
    ELISE_COPY(aMasqResul.border(1),0,aMasqResul.out());
    return aTRes._the_im;
}

Im2D_REAL4 ProlongByCont
     (
          Im2D_Bits<1> & aMasqRes,
          Im2D_Bits<1> aIMasq,
          Im2D_INT2 aInput,
          INT aNbProl,
          double aDistAdd,
          double aDistMaxAdd
     )   
{
     return TplProlongByCont(aMasqRes,aIMasq,aInput,aNbProl,aDistAdd,aDistMaxAdd);
}

Im2D_REAL4 ProlongByCont
     (
          Im2D_Bits<1> & aMasqRes,
          Im2D_Bits<1> aIMasq,
          Im2D_REAL4 aInput,
          INT aNbProl,
          double aDistAdd,
          double aDistMaxAdd
     )   
{
     return TplProlongByCont(aMasqRes,aIMasq,aInput,aNbProl,aDistAdd,aDistMaxAdd);
}





/*

Im2D_REAL4 ReduceImageProf(Im2D_Bits<1>,Im2D_REAL4 aImProf, const Box2dr &aBox,double aScale,Im2D_REAL4 aImPds,std::vector<Im2DGen*>  aVNew,std::vector<Im2DGen*> aVOld)

{
   return TplFReduceImageProf(TIm2DBits<1>(aIMasq),TIm2D<REAL4,REAL8>(aImProf),aBox,aScale,aVNew,aVOld);
}


Im2D_REAL4 ReduceImageProf(Im2D_REAL4 aImPds,Im2D_Bits<1> aIMasq,Im2D_INT2 aImProf,double aScale)
{
   return TplFReduceImageProf(aImPds,TIm2DBits<1>(aIMasq),TIm2D<INT2,INT>(aImProf),aScale);
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
