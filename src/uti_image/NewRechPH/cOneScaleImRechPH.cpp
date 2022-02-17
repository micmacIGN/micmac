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


#include "NewRechPH.h"



/*****************************************************/
/*                                                   */
/*            Constructeur                           */
/*                                                   */
/*****************************************************/


cOneScaleImRechPH::cOneScaleImRechPH(cAppli_NewRechPH &anAppli,const Pt2di & aSz,const double & aScale,const int & aNiv,int aPowDecim) :
   mAppli (anAppli),
   mSz    (aSz),
   mIm    (aSz.x,aSz.y),
   mTIm   (mIm),
   mImMod (1,1),
   mTImMod (mImMod),
   mScaleAbs (aScale),
   mScalePix (mScaleAbs/aPowDecim),
   mPowDecim (aPowDecim),
   mNiv      (aNiv),
   mNbPByLab   (int(eTPR_NoLabel),0),
   mSifDifMade (false)
{
   // cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
   // mInterp = new cTabIM2D_FromIm2D<tElNewRechPH>(aSinC,1000,false);
   mInterp = mAppli.Interp();

   mVoisGauss = SortedVoisinDisk(-1,2*mScalePix + 4,true);
   for (const auto & aPt : mVoisGauss)
   {
       mGaussVal.push_back(Gauss(mScalePix,euclid(aPt)));
   }
}



double cOneScaleImRechPH::QualityScaleCorrel(const Pt2di & aP0,int aSign,bool ImInit)
{
   tTImNRPH & aIm = ImInit ? mTIm : mTImMod;
   RMat_Inertie aMat;
   double aDef = -1e30;
   for (int aKV=0 ; aKV<int(mVoisGauss.size()) ; aKV++)
   {
        double aVal = aIm.get(aP0+mVoisGauss[aKV],aDef);

        if (aVal != aDef)
        {
           double aGausVal = mGaussVal.at(aKV);
           // Pour l'instant, un peu au pif, on met le poids egal a la gaussienne, 
           aMat.add_pt_en_place(aVal,aGausVal,aGausVal);
        }
   }
   mQualScaleCorrel = aSign * aMat.correlation();

   return mQualScaleCorrel;
}

void cOneScaleImRechPH::InitImMod()
{
  mImMod.Resize(mSz);
  mTImMod = tTImNRPH(mImMod);
}


cOneScaleImRechPH* cOneScaleImRechPH::FromFile
(
     cAppli_NewRechPH & anAppli,
     const double & aS0,
     const std::string & aName,
     const Pt2di & aP0,
     const Pt2di & aP1Init
)
{
   Tiff_Im aTifF = Tiff_Im::StdConvGen(aName,1,true);
   Pt2di aP1 = (aP1Init.x > 0) ? aP1Init : aTifF.sz();
   Pt2di aSzIm = aP1-aP0;
   cOneScaleImRechPH * aRes = new cOneScaleImRechPH(anAppli,aSzIm,aS0,0,1);

   if (anAppli.TestDirac())
   {
        ELISE_COPY(aRes->mIm.all_pts(),0,aRes->mIm.out());
        aRes->mTIm.oset((aP1-aP0)/2,1000);

        Tiff_Im::CreateFromIm(aRes->mIm,"LA-Im0.tif");
   }
   else
   {
      Im2D_REAL4 aImFl(aSzIm.x,aSzIm.y);

      double aVMax,aVMin;
      ELISE_COPY (aRes->mIm.all_pts(),trans(aTifF.in_proj(),aP0),aImFl.out() | VMax(aVMax) | VMin(aVMin) );

      double aMoy = (aVMax+aVMin) / 2;
      double aDyn = 3e4 / (aVMax-aVMin) ; // Ca sature a 32xxx ca laisse du rab

      ELISE_COPY ( aRes->mIm.all_pts(),(aImFl.in()-aMoy)*aDyn,aRes->mIm.out());
   }

   std::cout << "S00000000000 = " << aS0 << "\n";
   FilterGaussProgr(aRes->mIm,aS0,1.0,4);

   if (anAppli.SaveIm())
      Tiff_Im::CreateFromIm(aRes->mIm,"LA-Gaus0.tif");
   return aRes;
}

cOneScaleImRechPH* cOneScaleImRechPH::FromScale
                   (
                         cAppli_NewRechPH & anAppli,
                         cOneScaleImRechPH & anIm,
                         const double & aSigma,
                         int       aScaldec, // Scale decimation
                         bool      JumpDecim

                    )
{
     Pt2di aSzRes =  anIm.mSz;
     if (JumpDecim)
     {
        aSzRes = aSzRes / 2;
     }

     cOneScaleImRechPH * aRes = new cOneScaleImRechPH(anAppli,aSzRes,aSigma,anIm.mNiv+1,aScaldec);

     // Pour reduire le temps de calcul, si deja plusieurs iters la fon de convol est le resultat
     // de plusieurs iters ...
     int aNbIter = 4;
     if (!anAppli.TestDirac())
     {
        if (aRes->mNiv==1) aNbIter = 3;
        else if (aRes->mNiv>=2) aNbIter = 2;

        // std::cout << "GaussByExp, NbIter=" << aNbIter << "\n";
     }

     double aScaleInit = anIm.mScalePix;
     // anIm.mIm.dup(aRes->mIm);
     if (JumpDecim)
     {
         Pt2di aP;
         for (aP.x=0 ; aP.x<aSzRes.x ; aP.x++)
         {
             for (aP.y=0 ; aP.y<aSzRes.y ; aP.y++)
             {
                 aRes->mTIm.oset(aP,anIm.mTIm.get(aP*2));
             }
         }
         aScaleInit /= 2;
     }
     else
     {
        aRes->mIm.dup(anIm.mIm);
     }
     // Passe au filtrage gaussien le sigma cible et le sigma actuel, il se debrouille ensuite
     // std::cout << "FILTRAGE " << aScaleInit << " => " << aRes->mScalePix << "\n";
     FilterGaussProgr(aRes->mIm,aRes->mScalePix,aScaleInit,aNbIter);


     if (anAppli.SaveIm())
     {
        Tiff_Im::CreateFromIm(aRes->mIm,"LA-Gaus"+ ToString(aRes->Niv()) + ".tif");
     }
   
     if (anAppli.TestDirac())
     {
        TestDist(anIm.mSz,aRes->mIm.in(),aSigma);
     }
     return aRes;
}

// Indique si tous les voisins se compare a la valeur cible aValCmp
// autrement dit est un max ou min local

bool   cOneScaleImRechPH::SelectVois(const Pt2di & aP,const std::vector<Pt2di> & aVVois,int aValCmp)
{
    tElNewRechPH aV0 =  mTIm.get(aP);
    tElNewRechPH aVDef = aV0 - aValCmp; // Si c'est Max, aValCmp=1, donc neutre
    for (int aKV=0 ; aKV<int(aVVois.size()) ; aKV++)
    {
        const Pt2di & aVois = aVVois[aKV];
        tElNewRechPH aV1 =  mTIm.get(aP+aVois,aVDef);
        if (CmpValAndDec(aV0,aV1,aVois) != aValCmp)
           return false;
    }
    return true;
}

// Appele par Sift-Like

bool   cOneScaleImRechPH::ScaleSelectVois(cOneScaleImRechPH *aI2,const Pt2di & aP,const std::vector<Pt2di> & aVVois,int aValCmp,double aFactMul)
{
     if (! mTImMod.inside(aP)) 
        return false;
     if (! aI2->mTImMod.inside(aP)) 
        return false;

    static Pt2di aP00(0,0);
    tElNewRechPH aV0 =  round_ni(mTImMod.get(aP)*aFactMul);
    tElNewRechPH aV2 =  aI2->mTImMod.get(aP);


    if (aV0== aV2)
    {
// if (! DebugNRPH) ca ne change rien ?
{
       int aCmp = (mScaleAbs<aI2->mScaleAbs) ? -1 : 1;
       if (aCmp != aValCmp)
          return false;
}
    }
    else
    {
       int aCmp = (aV0<aV2) ? -1 : 1;
       if (aCmp != aValCmp)
          return false;
    }

    tElNewRechPH aVDef = aV0 -  aValCmp;

    for (int aKV=0 ; aKV<int(aVVois.size()) ; aKV++)
    {
        const Pt2di & aVois = aVVois[aKV];
        tElNewRechPH aV1 =  aI2->mTImMod.get(aP+aVois,aVDef);
        if (CmpValAndDec(aV0,aV1,aVois) != aValCmp)
           return false;
    }

    return true;
}


bool cOneScaleImRechPH::IsCol(int aRhoMax,const  std::vector<std::vector<Pt2di> >  & aVVPt,const Pt2di & aP)
{
   if ((aP.x <=aRhoMax) || (aP.y<=aRhoMax) || (aP.x>=mSz.x-aRhoMax-1) || (aP.y>=mSz.y-aRhoMax-1))
      return false;

   int aV0 = mTIm.get(aP);
   for (const auto & aVpt :  aVVPt)
   {
       std::vector<int> aVB;
       for (const auto aVois : aVpt)
       {
           aVB.push_back(CmpValAndDec(aV0,mTIm.get(aP+aVois),aVois));
       }
       aVB.push_back(aVB[0]);
       int aNbTrans = 0;
       for (int aK=1 ; aK<int(aVB.size()) ; aK++)
          if (aVB.at(aK-1) != aVB.at(aK))
             aNbTrans ++;

      if (aNbTrans!=2) 
         return false;
   }

   return true;
}



// Recherche tous les points topologiquement interessant

void cOneScaleImRechPH::CalcPtsCarac(bool Basic)
{
   std::vector<std::vector<Pt2di> > aVVPts;
   int aRhoMaxCol=0;
   {
      for (double aRho = 1.5; aRho < 3.0 * mScalePix  ; aRho+= 0.7)
      {
          cFastCriterCompute * aFCC = cFastCriterCompute::Circle(aRho);

          aVVPts.push_back(aFCC->VPt());
          delete aFCC;
          aRhoMaxCol = ElMax(round_up(aRho+1.5),aRhoMaxCol);
      }
   }

   static bool First = true;
   U_INT1  NbTrans[256];
   if (First)
   {
       for (int aFlag=0 ; aFlag<256 ; aFlag++)
       {
           NbTrans[aFlag] = 0;
           for (int aB=0 ; aB<8 ; aB++)
           {
               int aNextB = (aB+1) % 8;
               bool In0 = ((aFlag & (1<<aB)) != 0);
               bool In1 = ((aFlag & (1<<aNextB)) != 0);
               if (In0 != In1)
                  NbTrans[aFlag]++;
           }
       }
   }
   First = false;
   // voisin tries excluant le pixel central, le tri permet normalement de
   // beneficier le plus rapidement possible d'une "coupe"
   // std::vector<Pt2di> aVoisMinMax  = SortedVoisinDisk(0.5,mAppli.DistMinMax(Basic),true);
   std::vector<Pt2di> aVoisMinMax  = SortedVoisinDisk(0.5,6*mScalePix + 0.5,true);

   int aNbMax = 0;
   int aNbMin = 0;
   int aNbCol = 0;
   int aNbMaxStab = 0;
   int aNbMinStab = 0;
   int aNbColStab = 0;

   bool DoMin = mAppli.DoMin();
   bool DoMax = mAppli.DoMax();

   Im2D_U_INT1 aIFlag = MakeFlagMontant(mIm);
   TIm2D<U_INT1,INT> aTF(aIFlag);
   Pt2di aP ;
   for (aP.x = 1 ; aP.x <mSz.x-1 ; aP.x++)
   {
       for (aP.y = 1 ; aP.y <mSz.y-1 ; aP.y++)
       {
           int aFlag = aTF.get(aP);
           eTypePtRemark aLab = eTPR_NoLabel;

// std::cout << "DDDDDDDDd " << mAppli.DistMinMax(Basic) << "\n";
           
           if (DoMax &&  (aFlag == 0) )
           {
               aNbMax++;
              // std::cout << "DAxx "<< DoMax << " " << aFlag << "\n";
               if (SelectVois(aP,aVoisMinMax,1))
               {
                   aLab = eTPR_GrayMax;
                   aNbMaxStab++;
               }
           }
           if (DoMin &&  (aFlag == 255) )
           {
               aNbMin++;
               // std::cout << "DInnn "<< DoMin << " " << aFlag << "\n";
               if (SelectVois(aP,aVoisMinMax,-1))
               {
                   aLab = eTPR_GrayMin;
                   aNbMinStab++;
               }
           }

           // Les cols donnent un mauvais score
           if (false && (NbTrans[aFlag] == 2))
           {
               aNbCol++;
               if (IsCol(aRhoMaxCol,aVVPts,aP))
               {
                   aLab = eTPR_GraySadl;
                   aNbColStab++;
               }
           }

           if (aLab!=eTPR_NoLabel)
           {
              mLIPM.push_back(new cPtRemark(Pt2dr(aP),aLab,mNiv));
           }
        }
   }

   // Ne pas supprimer, stat utile sur les cols
   if (0)
   {
      std::cout << "  ====================  "
                << " NbMin " << aNbMin  << " " << aNbMinStab/double(aNbMin)
                << " NbMax " << aNbMax  << " " << aNbMaxStab/double(aNbMax)
                << " NbCol " << aNbCol  << " " << aNbMaxStab/double(aNbCol)
                << "\n";
   }

}

Pt3dr cOneScaleImRechPH::PtPly(const cPtRemark & aPRem,int aNiv)
{
   Pt2dr aPt = aPRem.RPtAbs(mAppli);
/*
{
  cOneScaleImRechPH *  aIm = mAppli.GetImOfNiv(aPRem.Niv());
  std::cout << "PPPPppp = " <<  aIm->PowDecim() << "\n";;
}
*/
   return Pt3dr(aPt.x,aPt.y,aNiv*mAppli.DZPlyLay());
}

void cOneScaleImRechPH::Export(cSetPCarac & aSPC,cPlyCloud *  aPlyC)
{
   mNbExLR = 0;
   mNbExHR = 0;
   static int aNbBr = 0;
   static int aNbBrOk = 0;

   for (std::list<cPtRemark*>::const_iterator itIPM=mLIPM.begin(); itIPM!=mLIPM.end() ; itIPM++)
   {
       cPtRemark & aP = **itIPM;
       double aDistZ = mAppli.DZPlyLay();

       // On est sur de regarder que les tetes d'arbres puisque on exige qu'il n'y ait pas de pere
       if (!aP.LowR())
       {
          cBrinPtRemark * aBr = new cBrinPtRemark(&aP,mAppli);
          aNbBr++;
          if (aBr->Ok())
          {
             aNbBrOk++;
             mAppli.AddBrin(aBr);

             cOnePCarac aPC;
             std::vector<cPtRemark *> aVP = aBr->GetAllPt();
             cPtRemark * aBif = aBr->Bifurk();
             if (aBif)
             {
                aPC.Pt() = aBif->RPt();
                int aS = SignOfType(aBif->Type());
                if (aS==1)  aPC.Kind() = eTPR_BifurqMax;
                if (aS==-1) aPC.Kind() = eTPR_BifurqMin;
                if (aS==0) aPC.Kind() = eTPR_BifurqSadl;
                aPC.DirMS() = Pt2dr(0,0);
             }
             else
             {
                  cPtRemark * aPMax =  aVP.at(aVP.size() - 1 -aBr->NivScal());
                  ELISE_ASSERT(aPMax->Niv()==aBr->NivScal(),"cOneScaleImRechPH::Export Incohe in Niv");

                  aPC.Pt() =  aPMax->RPt();
                  aPC.Kind() =  aVP.at(0)->Type();
                  aPC.DirMS() = aVP.back()->RPtAbs(mAppli) - aVP.front()->RPtAbs(mAppli);
             }

             aPC.Scale() = aBr->Scale();
             aPC.ScaleStab() = aBr->BrScaleStab();
             aPC.ScaleNature() = aBr->ScaleNature();

             aPC.NivScale() = aBr->NivScal();
             aPC.Id() = mAppli.GetNexIdPts();

             // std::cout << "DIRMS " << euclid(aPC.DirMS()) << "\n";

             aSPC.OnePCarac().push_back(aPC);
             if (aPlyC)
             {
                std::vector<cPtRemark *>  aVPt = aBr->GetAllPt();
                Pt3di aCol(NRrandom3(255),NRrandom3(255),NRrandom3(255));

                for (const auto & aPt : aVPt)
                {
                     aPlyC->AddSphere(aCol,PtPly(*aPt,aPt->Niv()),aDistZ/6.0,3);
                }
             }
          }
          else
          {
          }
       }
 
       if (aP.HighRs().empty()) mNbExHR ++;
       if (!aP.LowR()) mNbExLR ++;
   }
   std::cout << " Nb Br = "<<  aNbBr  << " perc Ok " << 100.0*( aNbBrOk/double(aNbBr))  << " OK=" << aNbBrOk << "\n";

   // std::cout << "NIV=" << mNiv << " HR " << mNbExHR << " LR " << mNbExLR << "\n";
}

void cOneScaleImRechPH::Show(Video_Win* aW)
{
   if (! aW) return;

   Im2D_U_INT1 aIR(mSz.x,mSz.y);
   Im2D_U_INT1 aIV(mSz.x,mSz.y);
   Im2D_U_INT1 aIB(mSz.x,mSz.y);


   ELISE_COPY(mIm.all_pts(),Max(0,Min(255,mIm.in())),aIR.out()|aIV.out()|aIB.out());

   for (std::list<cPtRemark*>::const_iterator itIPM=mLIPM.begin(); itIPM!=mLIPM.end() ; itIPM++)
   {
       Pt3di aC = Ply_CoulOfType((*itIPM)->Type(),0,1000);

       ELISE_COPY
       (
          disc((*itIPM)->RPt(),2.0),
          Virgule(aC.x,aC.y,aC.z),
          Virgule(aIR.oclip(),aIV.oclip(),aIB.oclip())
       );
   }

   ELISE_COPY
   (
      mIm.all_pts(),
      Virgule(aIR.in(),aIV.in(),aIB.in()),
      aW->orgb()
   );
}

// Initialise la matrice des pt remarquable, en Init les met, sinon les supprime
void cOneScaleImRechPH::InitBuf(const eTypePtRemark & aType, bool Init)
{
   for (std::list<cPtRemark *>::iterator itP=mLIPM.begin() ; itP!=mLIPM.end() ; itP++)
   {
      if ((*itP)->Type()==aType)
      {
         //  Buf2
         if (Init)
            mAppli.AddBuf2(*itP);
         else
            mAppli.ClearBuff2(*itP);
      }
   }
}


void cOneScaleImRechPH::CreateLink(cOneScaleImRechPH & aHR,const eTypePtRemark & aType)
{
   int aRatio = mPowDecim / aHR.mPowDecim;
   cOneScaleImRechPH & aLR = *this;

   // Initialise la matrice haute resolution
   aLR.InitBuf(aType,true);
   double aDist = (mScalePix * 6.0 + 0.5) ;

  
   int aNbHRTot=0;
   int aNbLinked=0;

   for (const auto & aPHR : aHR.mLIPM)
   {
       if (aPHR->Type()==aType)
       {
           aNbHRTot++;
           // Pt2di aPi = round_ni((*itP)->Pt());
           // tPtrPtRemark aPLR  =  mAppli.NearestPoint(round_ni(aPHR->Pt())/aRatio,aDist);
           tPtrPtRemark aPLR2 =  mAppli.NearestPoint2(aPHR->RPt()/aRatio,aDist);

           if (aPLR2)
           {
              // std::cout << "LEVS " << mNiv << " " << aHR.mNiv << "\n";
              aPLR2->MakeLink(aPHR);
              aNbLinked++;
           }
           else
           {
           }
       }
   }

/////
   if (0) // (aType==eTPR_GrayMax)
   {
      int aNbLRTot=0;
      for (const auto & aPHR : aLR.mLIPM)
      {
         if (aPHR->Type()==aType)
         {
             aNbLRTot++;
         }
           // Pt2di aPi = round_ni((*itP)->Pt());
      }

      if (aNbHRTot)
      {
         std::cout << "LLKKkkkk " << aNbHRTot << " " << aNbLinked  
                   << " PropH=" << aNbLinked/double(aNbHRTot)
                   << " PropHL=" << aNbLRTot/double(aNbHRTot) 
                   << " Ratio=" << aRatio
                   << " SP=" << mScalePix
                   << "\n";
      }
   }
   
   // Desinitalise 
   aLR.InitBuf(aType,false);
}

void cOneScaleImRechPH::CreateLink(cOneScaleImRechPH & aHR)
{
    for (int aK=0 ; aK<eTPR_NoLabel ; aK++)
    {
       CreateLink(aHR,eTypePtRemark(aK));
    }

   // std::cout <<  "CREATE LNK " << mNiv << "\n";
}


const int &  cOneScaleImRechPH::NbExLR() const {return mNbExLR;}
const int &  cOneScaleImRechPH::NbExHR() const {return mNbExHR;}
const double &  cOneScaleImRechPH::ScaleAbs() const {return mScaleAbs;}
const double &  cOneScaleImRechPH::ScalePix() const {return mScalePix;}
const int & cOneScaleImRechPH::PowDecim() const {return mPowDecim;}
bool cOneScaleImRechPH::SifDifMade() const {return mSifDifMade;}

bool cOneScaleImRechPH::SameDecim(const cOneScaleImRechPH & anI2) const
{
   return mPowDecim == anI2.mPowDecim;
}

cOneScaleImRechPH * cOneScaleImRechPH::GetSameDecimSiftMade(cOneScaleImRechPH* aI1,cOneScaleImRechPH* aI2)
{
   if (! mSifDifMade) return 0;
   if (SameDecim(*aI1) && aI1->mSifDifMade) return aI1;
   if (SameDecim(*aI2) && aI2->mSifDifMade) return aI2;

   return 0;
}


double cOneScaleImRechPH::GetVal(const Pt2di & aP,bool & Ok) const 
{
   double aRes = 0;
   Ok = mTIm.inside(aP);
   if (Ok) aRes = mTIm.get(aP);
   return aRes;
}


bool cOneScaleImRechPH::ComputeDirAC(cOnePCarac & aP)
{
   aP.OK() = true;

   cNH_CutAutoCorrelDir<tTImNRPH>  mACD(mTIm,Pt2di(aP.Pt()),ElMax(2.0,1+mScalePix),round_up(mScalePix));
   
    
   mACD.AutoCorrel(Pt2di(aP.Pt()),2.0);
   if (!  mACD.ResComputed())
   {
      aP.OK() =  false;
      return false;
   }
   Pt2dr  aR = mACD.Res();

   
   aP.AutoCorrel() = aR.y;
   aP.DirAC() = Pt2dr::FromPolar(mScalePix,aR.x);

   // std::cout << "AUTOC " << aP.AutoCorrel() << " " <<  mAppli.SeuilAC() << "\n";

   if (aP.AutoCorrel() > mAppli.SeuilAC())
   {
      aP.OK() =  false;
      return false;
   }

   // std::cout << "CALCUL ComputeDirAC " << isAC  << " " << aR<< "\n";
   return true;
}


bool    cOneScaleImRechPH::AffinePosition(cOnePCarac & aPt)
{

    Pt2dr aPt0 = aPt.Pt();
    aPt.Pt0() = aPt0;
    int aNbGrid = 2;
    double aStep = mScalePix/2;

    // On verifie on est dedans
    {
        double aSz = mInterp->SzKernel() *  aNbGrid * aStep;
        Pt2dr aPSz(aSz,aSz);
        int aRab = 3;
        Pt2di aPRab(aRab,aRab);

        Pt2di aP0 = round_down(aPt.Pt() -aPSz) - aPRab;
        Pt2di aP1 = round_up(aPt.Pt() +aPSz) + aPRab;
        if ((aP0.x<=0) || (aP0.y<=0) || (aP1.x>=mSz.x)|| (aP1.y>=mSz.y))
        {
            aPt.OK() = false;
            return false;
        }
    }

    tElNewRechPH ** aData = mIm.data();
    if ((aPt.Kind() == eTPR_LaplMax) || (aPt.Kind() == eTPR_LaplMin))
       aData = mImMod.data();

    L2SysSurResol aSys(6);

    double aCoeff[6];
    for (int aKx =-aNbGrid ; aKx<= aNbGrid ; aKx++)
    {
        for (int aKy =-aNbGrid ; aKy<= aNbGrid ; aKy++)
        {
            double aDx = aKx * aStep;
            double aDy = aKy * aStep;
            
            double aVal =   mInterp->GetVal(aData,aPt0 + Pt2dr(aDx,aDy));
            aCoeff[0] = 1.0;
            aCoeff[1] = aDx;
            aCoeff[2] = aDy;
            aCoeff[3] = aDx * aDx;
            aCoeff[4] = aDx * aDy;
            aCoeff[5] = aDy * aDy;
            
            aSys.AddEquation(1.0,aCoeff,aVal);
        }
    }

    Im1D_REAL8  aSol = aSys.Solve(&aPt.OK());
    if (! aPt.OK())
       return false;
    double * aDS = aSol.data();
    aDS[1] *= -0.5;
    aDS[2] *= -0.5;
    aDS[4] *= 0.5;
    // D0 + D1X + D2Y  + D3XX +  D4XY + D5YY+ 
    //  d4 = D4/2    d1 = -D1/2   d2 = -D2/2
    //   (D3    d4) X               (X)
    //   (d4    D5) Y   - 2 (d1 d2) (Y)

     double aDelta = aDS[3]*aDS[5] - ElSquare(aDS[4]);
     double X = ( aDS[5] * aDS[1] - aDS[4] * aDS[2]) / aDelta;
     double Y = (-aDS[4] * aDS[1] + aDS[3] * aDS[2]) / aDelta;

     // Two VP with different signs, will see when add sadle points
     if (aDelta<0)
     {
         aPt.OK() = false;
         return false;
     }

     // Verif solution est OK
     if (0)
     {
         std::cout << "CHhhhhhhhhhhhhhexxkkkk\n";
         int aS =  ((aDS[3]+aDS[5]) >0) ? 1 : -1; // SignOfType(aPt.Kind());

         Im1D_REAL8  aSol = aSys.Solve(&aPt.OK());
         double * aDS = aSol.data();

         for (int aK=0 ; aK<10 ; aK++)
         {
            double aDx = X + NRrandC() * mScalePix / 20.0;
            double aDy = Y + NRrandC() * mScalePix / 20.0;
            double aV =  (aDS[1]*aDx + aDS[2]*aDy + aDS[3]*aDx*aDx + aDS[4]*aDx*aDy + aDS[5]*aDy*aDy);
            double aVm = (aDS[1]*X + aDS[2]*Y + aDS[3]*X*X + aDS[4]*X*Y + aDS[5]*Y*Y);
            double aCheck = (aV-aVm) * aS;
        
            if (aCheck<0)
            {
               std::cout << " Vvvvv " << aV -aVm << " S=" << aS << "\n";
               ELISE_ASSERT(false,"cOneScaleImRechPH::AffinePosition Check min quad");
            }
         }
     }

     Pt2dr aCor(X,Y);
     if (euclid(aCor) > mScalePix)
     {
        return (aPt.OK() = false);
     }
     // std::cout << "Afffffinne Pos " << aPt.Pt()  << " " << aPt.Pt() + aCor << "\n";
     aPt.Pt() = (aPt.Pt() + aCor)* double(mPowDecim);

     return true;
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
aooter-MicMac-eLiSe-25/06/2007*/
