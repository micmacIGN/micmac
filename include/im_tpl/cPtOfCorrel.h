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

#ifndef _CPTOFCORREL_H_
#define _CPTOFCORREL_H_

// Classe pour calculer des points "favorables a la correlation 2D",
//  c'est a dire des point qui n'auto-correlent pas avec leur
// translate et qui ont du contraste

class cQAC_XYV;
class cQAC_Stat_XYV;
class cPtOfCorrel
{
    public :
        typedef cQAC_XYV tElem;
        typedef cQAC_Stat_XYV tCumul;


        cPtOfCorrel(Pt2di aSz,Fonc_Num aF,int aVCor);

        void NoOp() {}
        double AutoCorTeta(Pt2di aP,double aTeta,double anEpsilon);
        double BrutaleAutoCor(Pt2di aP,int aNbTeta,double anEpsilon);


        void QuickAutoCor_Gen(Pt2di aP,double & aLambda,double & aSvv);

        double QuickAutoCor(Pt2di aP);
        double QuickAutoCorWithNoise(Pt2di aP,double aBr);

        void MakeScoreAndMasq
             (
                   Im2D_REAL4 & aISc,
                   double  aEstBr,
                   Im2D_Bits<1> & aMasq,
                   double aSeuilVp,
                   double aSeuilEcart
             );


        void Init(const std::complex<int> & aP,cQAC_XYV &);
        void UseAggreg(const std::complex<int> & aP,const cQAC_Stat_XYV &);
        void OnNewLine(int);
   private :
       double  PdsCorrel(double anX,double anY);

       Pt2di                   mSz;
       int                     mVCor;
       Im2D_REAL4              mImIn;
       TIm2D<REAL4,REAL8>      mTIn;
       Im2D_REAL4              mGrX;
       TIm2D<REAL4,REAL8>      mTGrX;
       Im2D_REAL4              mGrY;
       TIm2D<REAL4,REAL8>      mTGrY;

       TIm2D<REAL4,REAL8> *   mISc;
       TIm2DBits<1> *         mMasq;
       double                 mEstBr;
       double                 mSeuilVP;
       double                 mSeuilEcart;
};

//  Classe pour reparti les points d'interet en respectant les criteres
//  suivants :
//       - pas de points dans le masque
//       - un point par cellule


class cRepartPtInteret
{
    public :
        cRepartPtInteret
        (
            Im2D_REAL4        aImSc,
            Im2D_Bits<1>      aMasq,
            const cEquiv1D &  anEqX,
            const cEquiv1D &  anEqY,
            double            aDistRejet,
            double            aDistEvitement
        );


        Im2D_Bits<1> ItereAndResult();

    private :
        void OnePasse(double aProp);

       void UpdatePond (Pt2di aP, Im2D_REAL4 aPond, bool Ajout);

       Pt2di    mNbC;
       Im2D_REAL4          mImSc;
       TIm2D<REAL4,REAL8>  mTSc;
       Im2D_Bits<1>        mMasq;
       TIm2DBits<1>        mTMasq;
       Pt2di               mSzGlob;
       Im2D_REAL4          mPondGlob;
       TIm2D<REAL4,REAL8>  mTPondGlob;

       cEquiv1D mEqX;
       cEquiv1D mEqY;
       Im2D_INT4 mPX;
       TIm2D<INT4,INT4> mTPx;
       Im2D_INT4 mPY;
       TIm2D<INT4,INT4> mTPy;
       double    mValRejet;
       double    mDRejMax;
       //double    mDRejCur;
       //double    mDEvCur;
       double    mDEvMax;
       Im2D_REAL4  mLastImPond;
};




#endif//  _CPTOFCORREL_H_

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
