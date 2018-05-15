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


cOneScaleImRechPH::cOneScaleImRechPH(cAppli_NewRechPH &anAppli,const Pt2di & aSz,const double & aScale,const int & aNiv) :
   mAppli (anAppli),
   mSz    (aSz),
   mIm    (aSz.x,aSz.y),
   mTIm   (mIm),
   mImMod (1,1),
   mTImMod (mImMod),
   mScale (aScale),
   mNiv   (aNiv),
   mNbPByLab (int(eTPR_NoLabel),0)
{
   // cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
   // mInterp = new cTabIM2D_FromIm2D<tElNewRechPH>(aSinC,1000,false);
   mInterp = mAppli.Interp();

   mVoisGauss = SortedVoisinDisk(-1,2*mScale + 4,true);
   for (const auto & aPt : mVoisGauss)
   {
       mGaussVal.push_back(Gauss(mScale,euclid(aPt)));
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
   cOneScaleImRechPH * aRes = new cOneScaleImRechPH(anAppli,aP1-aP0,aS0,0);

   if (anAppli.TestDirac())
   {
        ELISE_COPY(aRes->mIm.all_pts(),0,aRes->mIm.out());
        aRes->mTIm.oset((aP1-aP0)/2,1000);

      Tiff_Im::CreateFromIm(aRes->mIm,"LA-Im0.tif");
   }
   else
   {
      ELISE_COPY ( aRes->mIm.all_pts(),trans(aTifF.in_proj(),aP0),aRes->mIm.out());
   }

   std::cout << "S00000000000 = " << aS0 << "\n";
   FilterGaussProgr(aRes->mIm,aS0,1.0,4);

   if (anAppli.TestDirac())
      Tiff_Im::CreateFromIm(aRes->mIm,"LA-Gaus0.tif");
   return aRes;
}

cOneScaleImRechPH* cOneScaleImRechPH::FromScale(cAppli_NewRechPH & anAppli,cOneScaleImRechPH & anIm,const double & aSigma)
{
     cOneScaleImRechPH * aRes = new cOneScaleImRechPH(anAppli,anIm.mSz,aSigma,anIm.mNiv+1);

     // Pour reduire le temps de calcul, si deja plusieurs iters la fon de convol est le resultat
     // de plusieurs iters ...
     int aNbIter = 4;
     if (!anAppli.TestDirac())
     {
        if (aRes->mNiv==1) aNbIter = 3;
        else if (aRes->mNiv>=2) aNbIter = 2;

        std::cout << "GaussByExp, NbIter=" << aNbIter << "\n";
     }

     // anIm.mIm.dup(aRes->mIm);
     aRes->mIm.dup(anIm.mIm);
     // Passe au filtrage gaussien le sigma cible et le sigma actuel, il se debrouille ensuite
     FilterGaussProgr(aRes->mIm,aSigma,anIm.mScale,aNbIter);


     if (anAppli.TestDirac())
     {
        tImNRPH aIm =  aRes->mIm;
        Tiff_Im::CreateFromIm(aIm,"LA-Gaus"+ ToString(aRes->Niv()) + ".tif");
        TestDist(anIm.mSz,aIm.in(),aSigma);
     }
     return aRes;
}

// Indique si tous les voisins se compare a la valeur cible aValCmp
// autrement dit est un max ou min local

bool   cOneScaleImRechPH::SelectVois(const Pt2di & aP,const std::vector<Pt2di> & aVVois,int aValCmp)
{
    tElNewRechPH aV0 =  mTIm.get(aP);
    for (int aKV=0 ; aKV<int(aVVois.size()) ; aKV++)
    {
        const Pt2di & aVois = aVVois[aKV];
        tElNewRechPH aV1 =  mTIm.get(aP+aVois,aV0);
        if (CmpValAndDec(aV0,aV1,aVois) != aValCmp)
           return false;
    }
    return true;
}


bool   cOneScaleImRechPH::ScaleSelectVois(cOneScaleImRechPH *aI2,const Pt2di & aP,const std::vector<Pt2di> & aVVois,int aValCmp)
{
     if (! mTImMod.inside(aP)) 
        return false;
     if (! aI2->mTImMod.inside(aP)) 
        return false;

    static Pt2di aP00(0,0);
    tElNewRechPH aV0 =  mTImMod.get(aP);
    tElNewRechPH aV2 =  aI2->mTImMod.get(aP);


    if (aV0== aV2)
    {
       int aCmp = (mScale<aI2->mScale) ? -1 : 1;
       if (aCmp != aValCmp)
          return false;
    }
    else
    {
       int aCmp = (aV0<aV2) ? -1 : 1;
       if (aCmp != aValCmp)
          return false;
    }

    for (int aKV=0 ; aKV<int(aVVois.size()) ; aKV++)
    {
        const Pt2di & aVois = aVVois[aKV];
        tElNewRechPH aV1 =  aI2->mTImMod.get(aP+aVois,aV0);
        if (CmpValAndDec(aV0,aV1,aVois) != aValCmp)
           return false;
    }

    return true;
}


// Recherche tous les points topologiquement interessant

void cOneScaleImRechPH::CalcPtsCarac(bool Basic)
{
   // voisin tries excluant le pixel central, le tri permet normalement de
   // beneficier le plus rapidement possible d'une "coupe"
   // std::vector<Pt2di> aVoisMinMax  = SortedVoisinDisk(0.5,mAppli.DistMinMax(Basic),true);
   std::vector<Pt2di> aVoisMinMax  = SortedVoisinDisk(0.5,2*mScale + 4,true);


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
           
           if (DoMax &&  (aFlag == 0)  && SelectVois(aP,aVoisMinMax,1))
           {
              // std::cout << "DAxx "<< DoMax << " " << aFlag << "\n";
               aLab = eTPR_GrayMax;
           }
           if (DoMin &&  (aFlag == 255) && SelectVois(aP,aVoisMinMax,-1))
           {
               // std::cout << "DInnn "<< DoMin << " " << aFlag << "\n";
               aLab = eTPR_GrayMin;
           }

           if (aLab!=eTPR_NoLabel)
           {
              mLIPM.push_back(new cPtRemark(Pt2dr(aP),aLab,mNiv));
           }
        }
   }

}

Pt3dr cOneScaleImRechPH::PtPly(const cPtRemark & aP,int aNiv)
{
   return Pt3dr(aP.Pt().x,aP.Pt().y,aNiv*mAppli.DZPlyLay());
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

       if (!aP.LR())
       {
          cBrinPtRemark * aBr = new cBrinPtRemark(&aP,mAppli);
          aNbBr++;
          if (aBr->Ok())
          {
             aNbBrOk++;
             mAppli.AddBrin(aBr);

             cOnePCarac aPC;
             std::vector<cPtRemark *> aVP = aBr->GetAllPt();
             cPtRemark * aPMax =  aVP.at(aVP.size() - 1 -aBr->NivScal());
             ELISE_ASSERT(aPMax->Niv()==aBr->NivScal(),"cOneScaleImRechPH::Export Incohe in Niv");

             aPC.Kind() =  aVP.at(0)->Type();
             aPC.Pt() =  aPMax->Pt();
             aPC.Scale() = aBr->Scale();
             aPC.NivScale() = aBr->NivScal();
             aPC.DirMS() = aVP.back()->Pt() - aVP.front()->Pt();
             aPC.ScaleStab() = aBr->ScaleStab();
             aPC.ScaleNature() = aBr->ScaleNature();

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
/*
             int aN0 = aBr->Niv0();
             int aL  = aBr->Long();
             Pt3di aCol = Ply_CoulOfType(aP.Type(),aN0,aL);
             Pt3di aColP0 = (aN0!=0)  ? Pt3di(0,255,0)  : aCol;
             aPlyC->AddSphere(aColP0,PtPly(*(aBr->P0()),aN0),aDistZ/6.0,3);

             aPlyC->AddSphere(aCol,PtPly(*(aBr->PLast()),aN0+aL),aDistZ/6.0,3);

             int aNiv = aN0;
             for (cPtRemark * aP = aBr->P0() ; aP->LR() ; aP = aP->LR())
             {
                 aPlyC->AddSeg(aCol,PtPly(*aP,aNiv),PtPly(*(aP->LR()),aNiv+1),20);
                 aNiv++;
             }
*/
             }
          }
          else
          {
          }
       }
 
       if (aP.HRs().empty()) mNbExHR ++;
       if (!aP.LR()) mNbExLR ++;
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
          disc((*itIPM)->Pt(),2.0),
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
         Pt2di aPi = round_ni((*itP)->Pt());
         if (mAppli.Inside(aPi))
         {
            mAppli.PtOfBuf(aPi) = Init ? *itP : 0;
         }
         else
         {
         }
      }
   }
}


void cOneScaleImRechPH::CreateLink(cOneScaleImRechPH & aHR,const eTypePtRemark & aType)
{
   cOneScaleImRechPH & aLR = *this;

   // Initialise la matrice haute resolution
   aLR.InitBuf(aType,true);
   double aDist = mScale * 1.5 + 4;

  
   for (const auto & aPHR : aHR.mLIPM)
   {
       if (aPHR->Type()==aType)
       {
           // Pt2di aPi = round_ni((*itP)->Pt());
           tPtrPtRemark aPLR  =  mAppli.NearestPoint(round_ni(aPHR->Pt()),aDist);
           if (aPLR)
           {
              // std::cout << "LEVS " << mNiv << " " << aHR.mNiv << "\n";
              aPLR->MakeLink(aPHR);
           }
           else
           {
           }
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
const double &  cOneScaleImRechPH::Scale() const {return mScale;}


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


   cNH_CutAutoCorrelDir<tTImNRPH>  mACD(mTIm,Pt2di(aP.Pt()),ElMax(2.0,1+mScale),round_up(mScale));
   
    
   mACD.AutoCorrel(Pt2di(aP.Pt()),2.0);
   if (!  mACD.ResComputed())
   {
      aP.OK() =  false;
      return false;
   }
   Pt2dr  aR = mACD.Res();

   
   aP.AutoCorrel() = aR.y;
   aP.DirAC() = Pt2dr::FromPolar(mScale,aR.x);

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
    double aStep = mScale/2;

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
            double aDx = X + NRrandC() * mScale / 20.0;
            double aDy = Y + NRrandC() * mScale / 20.0;
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
     if (euclid(aCor) > mScale)
     {
        return (aPt.OK() = false);
     }
     aPt.Pt() = aPt.Pt() + aCor;

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
