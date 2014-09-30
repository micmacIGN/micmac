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
/*eLiSe06/05/99  
 
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY
	  
	    eLiSe : Elements of a Linux Image Software Environment
		 
		This program is free software; you can redistribute it and/or modify
		it under the terms of the GNU General Public License as published by
		the Free Software Foundation; either version 2 of the License, or
		(at your option) any later version.
		 
		This program is distributed in the hope that it will be useful,
		but WITHOUT ANY WARRANTY; without even the implied warranty of
		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
		GNU General Public License for more details.
		 
		You should have received a copy of the GNU General Public License
		along with this program; if not, write to the Free Software
		Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
		 
		  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS
		  Internet: Marc.Pierrot-Deseilligny@ign.fr
		     Phone: (33) 01 43 98 81 28              
*/

#ifndef _ELISE_GENERAL_CORREL_H_
#define _ELISE_GENERAL_CORREL_H_

class EliseCorrelation  // + ou - name space
{
    public :

	 // Permet de rajouter du flou dans les images
          static Fonc_Num FoncLissee(Fonc_Num ,Pt2di aSz,REAL  FactLissage,INT aNbStep,bool aUseVou,INT  aVout);

         // Pour rechecher au hasard, en aveugle une translation approximative

         static  Pt2di RechTransMaxCorrel
                            (
                                  Fonc_Num f1,
                                  Fonc_Num f2,
                                  Pt2di aSz,
                                  REAL  aRatioBord,
                                  REAL  FactLissage,
			          INT   aNbStepLiss = 2,
			          REAL   anEpsilon = 1e-5,
                                  bool   aUseVout = false,
                                  INT    aVout    = 0
                            );

         static  Im2D_REAL8 ImCorrelComplete
                            (
                                  Fonc_Num f1,
                                  Fonc_Num f2,
                                  Pt2di aSz,
                                  REAL  aRatioBord,
                                  REAL  FactLissage,
			          INT   aNbStepLiss = 2,
			          REAL   anEpsilon = 1e-5,
                                  bool   aUseVout = false,
                                  INT    aVout    = 0
                            );

          static Im2D_REAL8 ImLissee(Fonc_Num ,Pt2di aSz,REAL  FactLissage,
                                       INT aNbStep,bool aUseVou,INT  aVout);
     private :
};


class TabuledCollecPt2di
{
    public :
         TabuledCollecPt2di(Pt2di aP0,Pt2di aP1,REAL aRatio = 0.1);
         INT  NbPres (Pt2di,INT aDef) const;
         void Add (Pt2di) ;
         void clear();

         typedef std::vector<Pt2di>::const_iterator tIterPts;

    private  :

       bool  InDom(Pt2di) const;

       typedef U_INT2      tElem;
       tElem &   Val(Pt2di);
       const tElem &   Val(Pt2di) const;

       Pt2di                mP0;
       Pt2di                mP1;
       Pt2di                mSz;
       Im2D<tElem,INT>      mIm;
       tElem **             mCpt;
       std::vector<Pt2di>   mVPts;
};




class EliseDecCor2D
{
	public :
           Fonc_Num In();
           Fonc_Num In(Pt2dr aDef);
           Output Out();
	   EliseDecCor2D(Pt2di aSz);

           Pt2di anIDec(Pt2di aP) const;
           Pt2dr RDec(Pt2di aP) const;
           void  SetDec(Pt2di aP,Pt2dr aVal);
	private :
	   typedef REAL4 tElem;

           Im2D<tElem,REAL>  mDecX;
           tElem **          mDataX;
           Im2D<tElem,REAL>  mDecY;
           tElem **          mDataY;
};



class EliseCorrel2D
{
    public :

	   typedef REAL4  tElem;
	   typedef U_INT1 tElInit;
	   typedef INT    tBaseElInit;
   // Calcule le deplacement qui envoie l'image 2 dans Image 1
           EliseCorrel2D
           (
                 Fonc_Num f1,
                 Fonc_Num f2,
                 Pt2di aSz,
                 INT   aSzVgn,
                 bool  aUseVOut,   // Faut-il utiliser le param aVOut comme valeur d'exclusion de la correl init
                 INT   aVOut,
                 bool  WithPreCompute,
                 bool  WithDec
           );

          void SetSzI12(Pt2di aSz);

          void InitFromSousResol(EliseCorrel2D &,INT aRatio);

         
         Pt2di TrInitFromScratch
               (
		     INT   aZoomOverRes,
                     REAL  aRatioBord,
                     REAL  FactLissage
               );

           void ComputeCorrelMax(Pt2di aDec,INT Incert);
           Pt2dr Homol(Pt2di aP) const;
           Fonc_Num DecIn();
           Output   DecOut();
           Box2di BoxOk(Pt2di aDec,INT anIncert);

           Fonc_Num CorrelMax();
           Fonc_Num Fonc1();
           Fonc_Num Fonc2();

           void RaffineFromVois     
                (
                   INT anIncert,
                   Pt2di aDecGlob,
                   INT aNbVoisRech,  // Nomb des voisin explore
                   INT  aSzExt,      // dilatation du vois
                   bool enParal
               );


            Im2D_REAL4  SubPixelaire(Pt2di aDecGlob,REAL aStepLim,INT aSzV);
            Im2D_REAL4  DiffSubPixelaire(Pt2di aDecGlob,INT aSzV);
            INT SzV() const;
            REAL Correl(Pt2di aPIm1,Pt2di aPIm2) const;

            REAL CorrelStdIDec(Pt2di aPIm1) const;

            bool WithPrec() const {return mWithPreCompute;}
            bool WithDec() const {return mWithDec;}

           Im2D<tElInit,tBaseElInit>  I1() { return mI1;}
           Im2D<tElInit,tBaseElInit>  I2() { return mI2;}

           void SetI1GeomRadiomI2();
           Im2D<tElem,REAL> I1GeomRadiomI2() {return mI1GeomRadiomI2;}


          bool InImage(Pt2di aP);
          bool OkIm1(Pt2di aP);
          bool OkIm2(Pt2di aP);

         

           void RaffineFromVois     
                (
                   INT anIncert,
                   Pt2di aDecGlob,  // Utilise pour : zone de recherche + optimiser,
                                    // en conjonction avec incert,  la taille du Set de point (TabuledCollecPt2di)
	           EliseDecCor2D  &  mDecOut,
	           const EliseDecCor2D  &  mDecIn,
                   INT aNbVoisRech,  
                   INT  aSzExt     
                );

           bool OKPtCorrel(Pt2di aP0) const;
           REAL CorrelBrute(Pt2di aPIm1,Pt2di aPIm2) const;


           void ComputeCorrel(Pt2di aDec);
           void ComputeICorrel
           (
                Im2D_INT4  aSom12,
                Im2D_INT4  aBuf,
                Im2D_REAL4 aRes,
                Pt2di      aP0I1,
                Pt2di      aP0I2,
                Pt2di      aSz
           );


           void SetFoncs(Fonc_Num f1,Fonc_Num f2);
           void SetSzVign(INT aSzV);
           Fonc_Num Moy(Fonc_Num );

           Pt2di      mSzIm;
           bool       mWithPreCompute;
           bool       mWithDec;
           Pt2di      mSzPrec;
           Pt2di      mP0Correl;
           Pt2di      mP1Correl;


           void InitImOk(Im2D_Bits<1>,Im2D<tElInit,tBaseElInit>);

	   EliseDecCor2D     mDec;
           Im2D<tElem,REAL>  mCorrelMax;

           Im2D<tElem,REAL>   mCorrel;
           Pt2di              mSzI12;
           Im2D<tElInit,tBaseElInit>  mI1;
           tElInit **         mDataI1;
           Pt2di              mSzImOk;
           Im2D_Bits<1>       mIsImOk1;

           Im2D<tElInit,tBaseElInit>  mI2;
           tElInit **         mDataI2;
           Im2D_Bits<1>       mIsImOk2;
           
           Im2D<tElem,REAL>  mS1;
           tElem **          mDataS1;
           Im2D<tElem,REAL>  mS2;
           tElem **          mDataS2;
           Im2D<tElem,REAL>  mS11;
           tElem **          mDataS11;
           Im2D<tElem,REAL>  mS22;
           tElem **          mDataS22;


  // im1 en geom 2 (compte tenu de la correlation) avec
  // correction affine de la radiometrie
           Im2D<tElem,REAL>  mI1GeomRadiomI2;
           tElem **                 mD1GRI2;
           


           INT        mSzV;
           INT        mNbVois;
           bool       mUseVOut;
           INT        mVOut;
};


void Somme_12_2_22
     (
          Im2D_INT4 aSom12,
          Im2D_INT4 aSom2,
          Im2D_INT4 aSom22,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Im2D_U_INT1 anIm2,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     );

void Somme__1_11
     (
          Im2D_INT4 aSom,
          Im2D_INT4 aSom1,
          Im2D_INT4 aSom11,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     );

void Somme_Sup0_1_11
     (
          Im2D_INT4 aSom0,
          Im2D_INT4 aSom1,
          Im2D_INT4 aSom11,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     );

void Somme__1_2_11_12_22
     (
          Im2D_INT4 aSom,
          Im2D_INT4 aSom1,
          Im2D_INT4 aSom2,
          Im2D_INT4 aSom11,
          Im2D_INT4 aSom12,
          Im2D_INT4 aSom22,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Im2D_U_INT1 anIm2,
          Im2D_U_INT1 aPond,
          Pt2di       aP0,
          Pt2di       aP1,
          INT         aNb
     );



void Somme_12
     (
          Im2D_INT4 aSom12,
          Im2D_INT4 aBuf,
          Im2D_U_INT1 anIm1,
          Im2D_U_INT1 anIm2,
          Pt2di     aP0,
          Pt2di     aP1,
          INT       aNb
     );





#endif // _ELISE_GENERAL_CORREL_H_

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
