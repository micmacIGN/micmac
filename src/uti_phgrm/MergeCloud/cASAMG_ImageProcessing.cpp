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


#include "MergeCloud.h"

//============== ACCESSEURS AU PARAM =========================

class cCCMaxNbStep  : public cCC_NoActionOnNewPt
{
    public  :
       cCCMaxNbStep (int aNbMaxStep) :
          mNbStep    (0),
          mNbMaxStep (aNbMaxStep)
       {
       }

       void  OnNewPt(const Pt2di & aP) { mVPts.push_back(aP); }

       void OnNewStep() { mNbStep++;}
       bool  StopCondStep() {return mNbStep>mNbMaxStep;}
       bool ValidePt(const Pt2di &){return true;}

    // public :
       std::vector<Pt2di> mVPts;
       int mNbStep;
       int mNbMaxStep;
};

//================= INCIDENCE EN IMAGE =============================


class cCalcPlanImage : public cCCMaxNbStep
{
    public  :
       cCalcPlanImage(int aNbMaxStep,L2SysSurResol & aSys,Im2DGen * aImProf) :
             cCCMaxNbStep (aNbMaxStep),
             mSys         (&aSys),
             mImProf      (aImProf)
       {
            mSys->Reset();
       }
       void  OnNewPt(const Pt2di & aP) ;

   // private : 
       L2SysSurResol * mSys;
       Im2DGen *       mImProf;
};

void  cCalcPlanImage::OnNewPt(const Pt2di & aP) 
{
   cCCMaxNbStep::OnNewPt(aP);
   //  A x + By + C  = Z 
   static double aCoeff[3];
   aCoeff[0] = aP.x;
   aCoeff[1] = aP.y;
   aCoeff[2] = 1.0;

   mSys->AddEquation(1.0,aCoeff,mImProf->GetR(aP));
}


void cASAMG::ComputeIncidGradProf()
{
   Im2DGen * aImProf = mStdN->ImProf();
   L2SysSurResol  aSys (3);
   Pt2di aP0;

   Im2D_Bits<1> aMasqTmp = ImMarqueurCC(mSz);
   TIm2DBits<1> aTMasqTmp(aMasqTmp);


   for (aP0.x=0 ; aP0.x<mSz.x ; aP0.x++)
   {
       for (aP0.y=0 ; aP0.y<mSz.y ; aP0.y++)
       {
            double Angle = 1.5;
            if (mTMasqN.get(aP0))
            {
                cCalcPlanImage  aCalcPl(CCDist(),aSys,aImProf);
                int aNb = OneZC(aP0,CCV4(),aTMasqTmp,1,0,mTMasqN,1,aCalcPl);
                ResetMarqueur(aTMasqTmp,aCalcPl.mVPts);
                if (aNb >= SeuimNbPtsCCDist())
                {

                    Im1D_REAL8 aSol = aSys.Solve((bool *)0);
                    double * aDS = aSol.data();
                    Pt2dr aGrad (aDS[0],aDS[1]);

                    Angle = euclid(aGrad);
                }
/*
*/
            }
            mTIncid.oset(aP0,ElMin(255,ElMax(0,round_ni(Angle*DynAng()))));
       }
   }
}




//===================== INCIDEENCE EN EUCLIDIEN =========================

class cCalcPlanEuclid : public cCCMaxNbStep
{
    public  :
       cCalcPlanEuclid(int aNbMaxStep,const cRawNuage & aRN) :
           cCCMaxNbStep (aNbMaxStep),
           mRN          (&aRN)
       {
       }

       void  OnNewPt(const Pt2di & aP) 
       {
             cCCMaxNbStep::OnNewPt(aP);
             Pt3dr aP3 = mRN->GetPt(aP);
             mVP3.push_back(aP3);
             
       }

    // public :
       const cRawNuage * mRN;
       std::vector<Pt3dr> mVP3;
};

 
void cASAMG::ComputeIncidAngle3D()
{
   cRawNuage   aRN = mStdN->GetRaw();
   Pt2di aP0;

   Im2D_Bits<1> aMasqTmp = ImMarqueurCC(mSz);
   TIm2DBits<1> aTMasqTmp(aMasqTmp);


   for (aP0.x=0 ; aP0.x<mSz.x ; aP0.x++)
   {
       for (aP0.y=0 ; aP0.y<mSz.y ; aP0.y++)
       {
            double Angle = 1.5;
            if (mTMasqN.get(aP0))
            {
                cCalcPlanEuclid  aCalcPl(CCDist(),aRN);
                int aNb = OneZC(aP0,CCV4(),aTMasqTmp,1,0,mTMasqN,1,aCalcPl);
                ResetMarqueur(aTMasqTmp,aCalcPl.mVPts);
                if (aNb >= SeuimNbPtsCCDist())
                {
                    cElPlan3D aPlan(aCalcPl.mVP3,0);
                    // ElSeg3D aSeg(aC0,mStdN->GetPt(aP0));
                    ElSeg3D aSeg =  mStdN->FaisceauFromIndex(Pt2dr(aP0));
                    double aScal = scal(aPlan.Norm(),aSeg.TgNormee());
                    if (aScal<0) aScal = - aScal;
                    Angle = acos(ElMin(1.0,ElMax(-1.0,aScal)));
                }
            }
            mTIncid.oset(aP0,ElMin(255,ElMax(0,round_ni(Angle*DynAng()))));
       }
   }
}

//===================== INCIDEENCE KLIPSCHITZ =========================


void cASAMG::ComputeIncidKLip(Fonc_Num fMasq,double aPenteInPixel,int aNumQual)
{
   double aDynPix = mStdN->DynProfInPixel();

   Fonc_Num   aFoncProf =  mStdN->ImProf()->in_proj()/aDynPix;


   double aPdsZAbsolute = 0.0;
   // if (MPD_MM())
   aPdsZAbsolute = 0.0;

   Im2D_REAL4 aImZAbs(1,1);
   if (aPdsZAbsolute)
   {
        // double aFoc = mStdN->ResolutionAngulaire();
        // std::cout << "FFFFF=" << aFoc  << " " << 1/aFoc << "\n";
         // Tiff_Im::CreateFromFonc("Pax.tif",mSz,aFoncProf,GenIm::real4);
         // std::cout << "cASAMG::ComputeIncidKLip \n";


        Pt2di aP1 = mSz/2;
        Pt2di aP2 = aP1 + Pt2di(1,0);
        ElSeg3D aSeg1 = mStdN->Capteur2RayTer(Pt2dr(aP1));
        ElSeg3D aSeg2 = mStdN->Capteur2RayTer(Pt2dr(aP2));

        double aDist = euclid(aSeg1.TgNormee()-aSeg2.TgNormee()); 
        double aFoc = 1/aDist;
        // std::cout << "FFFFF=" << aDist  << " " << 1/aDist << "\n";

         
        aImZAbs.Resize(mSz);
       
        Pt2di aP; 
        for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
        {
            for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
            {
                if (mTMasqN.get(aP))
                {
                    double aProf = mStdN->ProfEuclidOfIndex(aP);
                    aProf = aFoc * log(ElMax(aProf,1e-30));
                    aImZAbs.SetR(aP,aProf);
                }
                else
                    aImZAbs.SetR(aP,0);
            }
        }

        ELISE_COPY
        (
             aImZAbs.interior(1),
             (1-aPdsZAbsolute ) * aFoncProf + aPdsZAbsolute * (-aImZAbs.in()),
             aImZAbs.out()
        );

        aFoncProf = aImZAbs.in_proj(); 
 
        // Tiff_Im::CreateFromFonc("ProfLog.tif",mSz,aImZAbs.in(),GenIm::real4);
        // getchar();
        // Tiff_Im::CreateFromFonc("Masq.tif",mSz,fMasq,GenIm::real4);

   }

   Fonc_Num  aOmbrStd = OmbrageKL( aFoncProf,fMasq,aPenteInPixel,2);
   Fonc_Num  aOmbrInv = OmbrageKL(-aFoncProf,fMasq,aPenteInPixel,1);
   Fonc_Num aOmbrGlob = Max(aOmbrInv,aOmbrStd);

   double aDynStore = 20.0;
   Im2D_U_INT1 aImLabel(mSz.x,mSz.y);
   TIm2D<U_INT1,INT> aTLab(aImLabel);
   Im2D_U_INT1 aImOmbr(mSz.x,mSz.y);
   ELISE_COPY(aImOmbr.all_pts(),Min(255,round_ni(aOmbrGlob*aDynStore)),aImOmbr.out());

   // 0 Out,  1 Ok,  2 Ok mais pentre forte, 3 voisin de 2, 4 retracte
   ELISE_COPY(aImLabel.all_pts(),fMasq + (aImOmbr.in()>0),aImLabel.out());
   ELISE_COPY(aImLabel.border(1),0,aImLabel.out());

   Im2D_Bits<1> aMarq = ImMarqueurCC(mSz);
   TIm2DBits<1> aTMarq (aMarq);


    // Supprime les CC de point "rouge" (=2) touchant du 0 et < 20 pts

    Pt2di aP;
    for (aP.x=1 ; aP.x<(mSz.x-1) ; aP.x++)
    {
        for (aP.y=1 ; aP.y<(mSz.y-1) ; aP.y++)
        {
            if (aTLab.get(aP)==2)
            {
                cCC_GetVPt aCC;
                OneZC(aP,true,aTMarq,1,0,aTLab,2,aCC);

                int aNbZ = (int)aCC.mVPts.size();
                bool ToSupr = (aNbZ < 20);

                for (int aKP=0; ToSupr && (aKP<aNbZ) ; aKP++)
                {
                    Pt2di aQ = aCC.mVPts[aKP];
                    for (int aKV=0 ; aKV<4 ; aKV++)
                    {
                         if (aTLab.get(aQ+TAB_4_NEIGH[aKV])==0)
                         {
                             ToSupr= false;
                         }
                    }
                }
                if (ToSupr) 
                {
                    for (int aKP=0;  aKP<int(aNbZ) ; aKP++)
                    {
                        aTLab.oset(aCC.mVPts[aKP],1);
                    }
                }
            }
        }
    }



   ELISE_COPY
   (
         select(aImLabel.all_pts(),NFoncDilatCond(aImLabel.in(0)==2,aImLabel.in(0)==1,true,2)&&(aImLabel.in()==1)),
         3,
         aImLabel.out()
   );
   ELISE_COPY
   (
         select(aImLabel.all_pts(),NFoncDilatCond(aImLabel.in(0)==1,aImLabel.in(0)==3,true,4)&&(aImLabel.in()==3)),
         4,
         aImLabel.out()
   );

   ELISE_COPY
   (
       select(aImLabel.all_pts(),(aImLabel.in()==2)||(aImLabel.in()==3)),
       aNumQual,
       mImQuality.out()
   );

}
/*

void cASAMG::ComputeIncidKLip(Im2D_Bits<1> aMasq,bool Inf,double aStep,int aSzVois)
{
   TIm2DBits<1> aTM(aMasq);
   Im2DGen * aImProf = mStdN->ImProf();
   int aSign = Inf ? 1 : -1;

   Pt2di aP0;
   for (aP0.x=0 ; aP0.x<mSz.x ; aP0.x++)
   {
       int aX0= ElMax(0,aP0.x-aSzVois);
       int aX1= ElMin(mSz.x-1,aP0.x+aSzVois);
       for (aP0.y=0 ; aP0.y<mSz.y ; aP0.y++)
       {
           if (aTM.get(aP0))
           {
              double aProf0 = aImProf->GetR(aP0);
              int aY0= ElMax(0,aP0.y-aSzVois);
              int aY1= ElMin(mSz.y-1,aP0.y+aSzVois);

              Pt2di aPV;
              for (aPV.x=aX0 ; aPV.x<=aX1 ; aPV.x++)
              {
                  for (aPV.y=aY0 ; aPV.y<=aY1 ; aPV.y++)
                  {
                      if (aTM.get(aPV))
                      {
                          double  aProf = aProf0 + aSign * euclid(
                      }
                  }
              }
           }
       }
   }
}
*/



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
