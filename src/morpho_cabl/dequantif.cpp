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



/**************************************************/
/*                                                */
/*            ElImplemDequantifier                */
/*                                                */
/**************************************************/

void ElImplemDequantifier::SetChamfer(const Chamfer & aChamf)
{
    mNbVYp = aChamf.nbv_yp();
    for (INT aK=0; aK<mNbVYp ; aK++)
    {
       Pt2di aP = aChamf.neigh_yp ()[aK];
       mVYp[aK]   = mSzReel.x * aP.y + aP.x;
       mPdsYp[aK] = aChamf.pds_yp ()[aK];
    }


    mNbVYm = aChamf.nbv_yn();
    for (INT aK=0; aK<mNbVYm ; aK++)
    {
       Pt2di aP = aChamf.neigh_yn ()[aK];
       mVYm[aK]   = mSzReel.x * aP.y + aP.x;
       mPdsYm[aK] = aChamf.pds_yn ()[aK];
    }

    mNbVYT = aChamf.nbv();
    for (INT aK=0; aK<mNbVYT ; aK++)
    {
       Pt2di aP = aChamf.neigh ()[aK];
       mVYT[aK]   = mSzReel.x * aP.y + aP.x;
       mPdsYT[aK] = aChamf.pds ()[aK];
    }
}


void ElImplemDequantifier::SetSize(Pt2di aSz)
{
    mSzReel = aSz + Pt2di(2,2);

    mImQuant.Resize(mSzReel);
    lDQ = mImQuant.data_lin();
    mDistPlus.Resize(mSzReel);
    mDPL = mDistPlus.data_lin();
    mDistMoins.Resize(mSzReel);
    mDM = mDistMoins.data_lin();
    mNbPts = mSzReel.x * mSzReel.y;

    mImDeq.Resize(aSz);
}

ElImplemDequantifier::ElImplemDequantifier(Pt2di aSz) :
  mSzReel    (aSz+Pt2di(2,2)),
  mTraitCuv  (false),
  mImQuant   (mSzReel.x,mSzReel.y),
  mDistPlus  (mSzReel.x,mSzReel.y),
  mDistMoins (mSzReel.x,mSzReel.y),
  mImDeq     (aSz.x,aSz.y)
{
}

void ElImplemDequantifier::SetTraitSpecialCuv(bool WithTrait)
{
   mTraitCuv = WithTrait;
}


void ElImplemDequantifier::OnePasse
     (
                  INT aPBegin, INT aStepP,INT aPend,
                  INT aNbV,INT * aTabV,INT * aTabP
     )
{
    for (INT aP0=aPBegin; aP0!=aPend ; aP0+=aStepP)
    {
        INT aQuant0 = lDQ[aP0];
        if (aQuant0 != eValOut)
        {
            for (INT aKV=0; aKV<aNbV ; aKV++)
            {
                INT aPV = aP0 + aTabV[aKV];
                // INT aQuantV = lDQ[aPV];
                if (lDQ[aPV] == aQuant0)
                {
                   INT aPds = aTabP[aKV];
                   ElSetMin(mDPL[aP0],mDPL[aPV]+aPds);
                   ElSetMin(mDM[aP0],mDM[aPV]+aPds);
                }
            }
        }
    }
}

void ElImplemDequantifier::TraitCuv ( U_INT2 *  aDA,U_INT2 *  aDB)
{

    static std::vector<INT> Pts;
    Pts.reserve(3000);


    for (INT aP0=0; aP0<mNbPts ; aP0++)
    {
        if ((aDA[aP0] == eMaxDist) && (lDQ[aP0] != eValOut) )
	{
            Pts.push_back(aP0);
	    aDA[aP0] = eMaxDist-1;
	    INT aK = 0;
	    INT  NbV = 0;
	    INT  SigmaDV = 0;
            INT  aDmaxB = 0;
	    while (aK != INT (Pts.size()))
	    {
		INT aP =  Pts[aK];


                ElSetMax(aDmaxB,aDB[aP]);
		for (INT aKV =0 ; aKV < mNbVYT ; aKV++)
		{
                    INT aPV = aP + mVYT[aKV];
		    if (lDQ[aPV] == eValOut)
                    {
                    }
		    else if (aDA[aPV] == eMaxDist)
                    {
                         Pts.push_back(aPV);
	                 aDA[aPV] = eMaxDist-1;
                    }
                    else if (aDA[aPV] < eMaxDist-1)
	            {
			    NbV++;
			    SigmaDV += aDB[aPV];
		    }
		}
                aK++;
	    }
	    REAL DMoy = SigmaDV/REAL( NbV);
	    for (aK =0 ; aK<INT(Pts.size()) ; aK++)
            {
                INT aP = Pts[aK];
                INT aD = aDB[aP];

                REAL aPds = (2*aD*aDmaxB - ElSquare(aD))/REAL (aDmaxB*(aDmaxB+DMoy));


                aDB[aP] = round_ni(aPds*10000);
                aDA[aP] = 10000 - aDB[aP];
            }
            Pts.clear();
	}
    }

}

void ElImplemDequantifier::OnePasseVideo()
{
    OnePasse(0,1,mNbPts,mNbVYm,mVYm,mPdsYm); 
}

void ElImplemDequantifier::OnePasseInverseVideo()
{
    OnePasse(mNbPts-1,-1,-1,mNbVYp,mVYp,mPdsYp); 
}


void ElImplemDequantifier::QuickSetDist(INT aNbStep)
{
    ELISE_COPY(mDistPlus.all_pts(),eMaxDist,mDistPlus.out());
    ELISE_COPY(mDistMoins.all_pts(),eMaxDist,mDistMoins.out());




    for (INT aP0=0; aP0<mNbPts ; aP0++)
    {
        INT aQuant0 = lDQ[aP0];
        if (aQuant0 != eValOut)
        {
            for (INT aKV=0; aKV<mNbVYm ; aKV++)
            {
                INT aPV = aP0 + mVYm[aKV];
                INT aQuantV = lDQ[aPV];
                if (aQuantV != eValOut)
                {
                   INT aPds = mPdsYm[aKV];
                   if (aQuantV<aQuant0)
                   {
                       ElSetMin(mDPL[aPV],aPds-1);
                       ElSetMin(mDM[aP0],aPds-1);
                   }
                   else if (aQuantV>aQuant0)
                   {
                       ElSetMin(mDPL[aP0],aPds-1);
                       ElSetMin(mDM[aPV],aPds-1);
                   }
                   else 
                   {
                      ElSetMin(mDPL[aP0],mDPL[aPV]+aPds);
                      ElSetMin(mDM[aP0],mDM[aPV]+aPds);
                   }
                }
            }
        }
    } 

    OnePasseInverseVideo();

    for (INT aK=0; aK<aNbStep ; aK++)
    {
       OnePasseVideo();
       OnePasseInverseVideo();
    }

    if (mTraitCuv)
    {
       TraitCuv(mDPL,mDM);
       TraitCuv(mDM ,mDPL);
    }
}

void ElImplemDequantifier::DoDequantif(Pt2di aSzIm,Fonc_Num f2Deq,bool aVerifI)
{
     DoDequantifWithMasq(aSzIm,f2Deq,Fonc_Num(0),aVerifI);
}

void ElImplemDequantifier::DoDequantifWithMasq(Pt2di aSzIm,Fonc_Num f2Deq,Fonc_Num FMasqOut,bool aVerifI)
{

     SetSize(aSzIm);
     SetChamfer(Chamfer::d32);


     Symb_FNum aFC = Max(trans(Rconv(f2Deq),Pt2di(-1,-1)),eValOut+2);
     Symb_FNum aIFC = round_ni(aFC);
     double aDifI;
     ELISE_COPY
     (
          mImQuant.interior(1),
          Virgule(aIFC,Abs(aFC-aIFC)),
          // Max(trans(f2Deq,Pt2di(-1,-1)),eValOut+2),
          Virgule(mImQuant.out(),VMax(aDifI))
     );
     if (aVerifI)
     {
        ELISE_ASSERT(aDifI<1e-6,"Non int in ElImplemDequantifier::DoDequantif");
     }

     ELISE_COPY(mImQuant.border(1),eValOut, mImQuant.out());
     ELISE_COPY(select(mImQuant.interior(1),trans( FMasqOut,Pt2di(-1,-1))), eValOut, mImQuant.out());



     QuickSetDist(1);
   
     Symb_FNum  sM (mDistMoins.in()); 
     Symb_FNum  sP (mDistPlus.in()); 

/*
Pt2di P128(1565+2,1100+2);
Pt2di P126(1475+2,1085+2);

std::cout <<   "P128 : " 
          << int(mDistMoins.data()[P128.y][P128.x]) << " " 
          << int(mDistPlus.data()[P128.y][P128.x]) << "\n" ;
std::cout <<   "P126 : " 
          << int(mDistMoins.data()[P126.y][P126.x]) << " " 
          << int(mDistPlus.data()[P126.y][P126.x]) << "\n" ;


std::cout << "SZ IM " <<  aSzIm << "\n"; getchar();
*/


     ELISE_COPY
     (
         mImDeq.all_pts(),
         trans(mImQuant.in()+(sM/Rconv(sM+sP)-0.5),Pt2di(1,1)),
         mImDeq.out()
     );
}


Fonc_Num ElImplemDequantifier::ImDeqReelle()
{
   return mImDeq.in_proj();
}

Fonc_Num ElImplemDequantifier::PartieFrac(INT anAmpl)
{
   Symb_FNum  sM (mDistMoins.in()); 
   Symb_FNum  sP (mDistPlus.in()); 

   return trans ( ((sM-sP)*anAmpl) / (2*(sM+sP)), Pt2di(1,1));
}

void VisuGray(Video_Win aW,Fonc_Num aFonc)
{
    REAL aMax,aMin;
    ELISE_COPY(aW.all_pts(),Rconv(aFonc),VMax(aMax)|VMin(aMin));
    ELISE_COPY(aW.all_pts(),(aFonc-aMin)*255.0/(aMax-aMin),aW.ogray());
}

/*
static Fonc_Num Moy(Fonc_Num aF,INT aNbV)
{
   return rect_som(aF,aNbV)/ElSquare(1.0+2*aNbV);
}
*/

void ElImplemDequantifier::Test()
{
    INT Z=2;

    Pt2di aSZ = mSzReel - Pt2di(2,2);

    Video_Win aW = Video_Win::WStd(aSZ,Z);
    aW.set_title("Image Quant");
    Video_Win aW2 = Video_Win::WStd(aSZ,Z);
    Video_Win aW3 = Video_Win::WStd(aSZ,Z);
    Video_Win aW4 = Video_Win::WStd(aSZ,Z);
    aW2.set_title("Deq");
    aW3.set_title("Deq+Cuv");
    aW4.set_title("FRELLE");


    Fonc_Num aFR = sin(FX/20.0) * sin(FY/20.0) * 3 +  FX/70.0;

// aFR = 5 * (1- (Square(FX-aSZ.x/2)+Square(FY-aSZ.y/2))/square_euclid(aSZ/2));

    Fonc_Num aFonc = round_ni (aFR);
                    

     Fonc_Num aBase = 0;// aFonc;


    REAL Ampl = 90.0;
    ELISE_COPY(aW.all_pts(),(aFonc -aBase)*Ampl,aW.ocirc());

    DoDequantif(aSZ,aFonc,true);



    ELISE_COPY ( aW2.all_pts(), (ImDeqReelle()-aBase)*Ampl, aW2.ocirc());
    
    SetTraitSpecialCuv(true);
    DoDequantif(aSZ,aFonc,true);
    ELISE_COPY ( aW3.all_pts(), (ImDeqReelle()-aBase)*Ampl, aW3.ocirc());
    ELISE_COPY ( aW4.all_pts(), (aFR-aBase)*Ampl, aW4.ocirc());

/*
    Video_Win aW4 = Video_Win::WStd(aSZ,Z);
    ELISE_COPY 
    ( 
          aW3.all_pts(), 
          Moy(Moy( aF.in(0),3),3) *Ampl,
          aW3.ocirc()
    );
    ELISE_COPY 
    ( 
          aW3.all_pts(), 
          Moy(Moy( ImDeqReelle(),3),3) *Ampl,
          aW4.ocirc()
    );

*/


    while(1) getchar();
}

void TEST_DEQ()
{
    // Tiff_Im aF("/home/pierrot/T.tif");
    // Tiff_Im aF("/data/AutoCalib/SimulNew/TmpId012ZPx_Im1Im2Reduc_4.tif");

    ElImplemDequantifier aDeq(Pt2di(200,200));
    aDeq.Test();
}



Im2D_U_INT2 ElImplemDequantifier::DistPlus()
{
   return mDistPlus;
}

Im2D_U_INT2 ElImplemDequantifier::DistMoins()
{
   return mDistMoins;
}

Im2D_REAL4 ReduceBin(Im2D_REAL4 aImIn)
{
    Im2D_REAL4 aRes = ReducItered(aImIn,1);
    ELISE_COPY(aRes.all_pts(),aRes.in()>0.99,aRes.out());

    return aRes;
}

Im2D_REAL4 RecursiveImpaint
     (
          Im2D_REAL4 aFlMaskInit,
          Im2D_REAL4 aFlMaskFinal,
          Im2D_REAL4 aFlIm,
          int        aDeZoom,
          int        aZoomCible
     )
{

    Pt2di aSz = aFlIm.sz();
    Im2D_REAL4 aSolInit(aSz.x,aSz.y);
    ELISE_COPY(aFlIm.all_pts(),aFlIm.in(),aSolInit.out());


    TIm2D<REAL4,REAL> aTMaskI(aFlMaskInit);

   int aNbIter = 2 + 3 * aDeZoom;

    if (aDeZoom >=aZoomCible)
    {
       aNbIter += ElSquare(aNbIter)/2;
       Im2D_REAL8 aDblMasqk(aSz.x,aSz.y);

       ELISE_COPY(aFlIm.all_pts(),aFlMaskInit.in(),aDblMasqk.out());
       // On met dans le masq uniquement les bords
       ELISE_COPY
       (
            select(aDblMasqk.all_pts(), erod_d4(aDblMasqk.in(0)>0.5,1)),
            0,
            aDblMasqk.out()
       );

       Im2D_REAL8 aDblIm(aSz.x,aSz.y);
       ELISE_COPY(aDblIm.all_pts(),aFlIm.in()*aDblMasqk.in(),aDblIm.out());

       FilterGauss(aDblIm,8.0,1);
       FilterGauss(aDblMasqk,8.0,1);

       ELISE_COPY
       (
           select(aSolInit.all_pts(),aFlMaskInit.in()<0.5),
           aDblIm.in() / Max(1e-20,aDblMasqk.in()),
           aSolInit.out()
       );

    }
    else
    {
        TIm2D<REAL4,REAL> aTSolInit(aSolInit);
        TIm2D<REAL4,REAL> aTIm(aFlIm);

        Im2D_REAL4 aSsEch = RecursiveImpaint
                            (
                                ReduceBin(aFlMaskInit),
                                ReduceBin(aFlMaskFinal),
                                ReducItered(aFlIm,1),
                                aDeZoom*2,
                                aZoomCible
                            );

         TIm2D<REAL4,REAL> aTSsE(aSsEch);
         Pt2di aP;
         for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
         {
             for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
             {
                 double aPdsI = aTMaskI.get(aP);
                 if (aPdsI <0.999)
                 {
                     double aVal =  aPdsI * aTIm.get(aP) 
                                    + (1-aPdsI) * aTSsE.getprojR(Pt2dr(aP.x/2.0,aP.y/2.0));
                      aTSolInit.oset(aP,aVal);
                 }
             }
         }
      }


      TIm2D<REAL4,REAL> aTMaskF(aFlMaskFinal);
      std::vector<Pt2di> aVF;
      {
          Pt2di aP;
          for (aP.x=1 ; aP.x<aSz.x-1 ; aP.x++)
          {
              for (aP.y=1 ; aP.y<aSz.y-1 ; aP.y++)
              {
                   if ((aTMaskI.get(aP)<0.999) && (aTMaskF.get(aP)>0.001))
                   {
                      aVF.push_back(aP);
                   }
              }
          }
      }


      int aNbPts = (int)aVF.size();
      for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
      {
          TIm2D<REAL4,REAL> aTSolInit(aSolInit);
          TIm2D<REAL4,REAL> aTIm(aFlIm);
          Im2D_REAL4        aNewSol (aSz.x,aSz.y);
          TIm2D<REAL4,REAL> aTNew(aNewSol);
          aNewSol.dup(aFlIm);

          for (int aKP=0 ; aKP<aNbPts ; aKP++)
          {
              Pt2di aP =  aVF[aKP];
              float aSomV=0;
              float aSomM=0;
              for (int aKV = 0 ; aKV<5 ; aKV++)
              {
                  Pt2di aPV = aP+ TAB_5_NEIGH[aKV];
                  float aM = (float)aTMaskF.get(aPV);
                  aSomM += aM;
                  aSomV += aM *(float)aTSolInit.get(aPV);
              }
              float aPdsI = (float)aTMaskI.get(aP);
              float aVal =  aPdsI * (float)aTIm.get(aP) + (1-aPdsI) * (aSomV/aSomM);
              aTNew.oset(aP,aVal);
              
          }

          aSolInit = aNewSol;
      }

      return aSolInit;
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
