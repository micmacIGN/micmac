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



#ifndef _EX_OPER_ASSOC_EXTERN_H_
#define _EX_OPER_ASSOC_EXTERN_H_

#include <iostream>
#include <math.h>

#include "im_tpl/oper_assoc_exter.h"

namespace NS_TestOpBuf
{
  // Classe definissant une interface minimale sur une
  // une image 2D, juste pour donner un exemple non trivial

typedef unsigned char tElImage;

class cInterfaceIm2D
{
    public :
        virtual tElImage GetValue(const std::complex<int> &) const = 0;
        virtual void     SetValue(const std::complex<int> &,const tElImage &) = 0;
        virtual ~cInterfaceIm2D() {};
    private :
};


    /*******************************************************/
    /*                                                     */
    /*       EXEMPLE CORREL                                */
    /*                                                     */
    /*******************************************************/


// Definie le type des entree necessaires au calcul
struct  cVal2Image
{
   // Aucun Prerequis pour cette classe
     tElImage mVIm1;
     tElImage mVIm2;
};

// Classe permettant de cumuler l'info

class cCumulVarCov
{
    public :
        // Constructeur sans Arg pour l'initialisation; Prerequis
        cCumulVarCov () :
            mNb    (0),
            mSom1  (0),
            mVar11 (0),
            mSom2  (0),
            mVar22 (0),
            mCov12 (0)
        {
        }

        // Operation pour cumuler deux  Variance/Covariance  ; Prerequis
        void AddCumul(int aSigne,const cCumulVarCov & aCVC)
        {
               mNb     += aSigne * aCVC.mNb;
               mSom1   += aSigne * aCVC.mSom1;
               mVar11  += aSigne * aCVC.mVar11;
               mSom2   += aSigne * aCVC.mSom2;
               mVar22  += aSigne * aCVC.mVar22;
               mCov12  += aSigne * aCVC.mCov12;
        }

        //  Operation pour ajouter une nouvelle entree a la Var/Cov ; Pre requis

        void AddElem(int aSigne,const cVal2Image & anEl)
        {
            mNb    += aSigne;
            mSom1  += aSigne * anEl.mVIm1;
            mVar11 += aSigne * anEl.mVIm1 * anEl.mVIm1;
            mSom2  += aSigne * anEl.mVIm2;
            mVar22 += aSigne * anEl.mVIm2 * anEl.mVIm2;
            mCov12 += aSigne * anEl.mVIm1 * anEl.mVIm2;
        }
    // private :

        int mNb;
        int mSom1;
        int mVar11;
        int mSom2;
        int mVar22;
        int mCov12;
};



// La classe qui va permettre d'instancier cTplOpbBufImage
// pour faire de la correlation rapide de 2 images

class cArgCorrelRapide2Im
{
    public :

       // Definition des types element d'entree et cumul : Prerequis
        typedef cVal2Image    tElem;
        typedef cCumulVarCov  tCumul;


        // Fonction qui sera appelee a chaque nouvelle ligne,
        // en general ne fait rien  : Prerequis
        void OnNewLine(int anY)
        {
             if (0) 
                std::cout << "J'en suis a la ligne " << anY << "\n";
        }

        // Fonction appelee , en "sortie" pour utiliser  le resultat
        // Prerequis
        void UseAggreg(const std::complex<int> & aP,const cCumulVarCov & aCVC)
        {
           // Formule classique pour calculer les moment d'ordre 2 centres
           // a partir des moments d'ordre 0, 1 et 2
            double aS1 = aCVC.mSom1   / (double) aCVC.mNb;
            double aS2 = aCVC.mSom2   / (double) aCVC.mNb;
            double aS11 = aCVC.mVar11 / (double) aCVC.mNb - aS1*aS1;
            double aS22 = aCVC.mVar22 / (double) aCVC.mNb - aS2*aS2;
            double aS12 = aCVC.mCov12  / (double) aCVC.mNb - aS1*aS2;

            double anEct = aS11 * aS22;
            // Evite / 0, voir racine negative
            //  si la variance est trop faible, 
            if (anEct < mEctMin)
               anEct = mEctMin;
            // aCoefCorr = Le coefficient de correlation compris entre -1 et 1 
            double aCoefCorr = aS12 / std::sqrt(anEct);

            mImOut.SetValue(aP,int((1+aCoefCorr)*100.0));
        }

        // Fonction appelee , en "entree" pour memoriser les entrees,
        // ici les entrees sont relativement triviales, PreRequis

        void Init(const std::complex<int> & aP,cVal2Image & anEl)
        {
            anEl.mVIm1 = mIm1.GetValue(aP);
            anEl.mVIm2 = mIm2.GetValue(aP);
        }


    // *********** FIN DES PRE-REQUIS

       cArgCorrelRapide2Im 
       (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const cInterfaceIm2D &   anIm2
       )  :
             mImOut  (anImOut),
             mIm1    (anIm1  ),
             mIm2    (anIm2  ),
             mEctMin (1e-5)
       {
       }


    private :

        cInterfaceIm2D &         mImOut;
        const cInterfaceIm2D &   mIm1;
        const cInterfaceIm2D &   mIm2;
        double                   mEctMin;

};


void TestCorrel
     (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const cInterfaceIm2D &   anIm2,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
           
     );

    /*******************************************************/
    /*                                                     */
    /*       EXEMPLE SOMME SIMPLE                          */
    /*                                                     */
    /*******************************************************/

class cCumulSomIm
{
    public :
         cCumulSomIm () :
              mSom (0)
         {
         }
         void AddCumul(int aSigne,const cCumulSomIm & aCSI)
         {
              mSom += aSigne * aCSI.mSom;
         }
         void AddElem(int aSigne,const int & anEl)
         {
              mSom += aSigne * anEl;
         }
         int Som() const {return mSom;}
    private :
         int mSom;

};

class cArgSommeRapide1Im
{
    public :

      // Definition des types element d'entree et cumul : Prerequis
        typedef tElImage     tElem;
        typedef cCumulSomIm  tCumul;
        void OnNewLine(int /*anY*/) {}

        void UseAggreg(const std::complex<int> & /*aP*/,const cCumulSomIm & /*aCSI*/)
        {
        };
        void Init(const std::complex<int> & aP,tElImage & anEl)
        {
            anEl = mImIn.GetValue(aP);
        }
        cArgSommeRapide1Im
        (
             const cInterfaceIm2D &   anImIn
        ) :
            mImIn (anImIn)
        {
        }
    private :
        const cInterfaceIm2D &   mImIn;
};


void TestSomRapide
     (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
           
     );





    /*******************************************************/
    /*                                                     */
    /*       EXEMPLE SOMME ITEREE                          */
    /*                                                     */
    /*******************************************************/

class cArgSommeRapideIteree
{
    public :

      // Definition des types element d'entree et cumul : Prerequis
        typedef int     tElem;
        typedef cCumulSomIm  tCumul;
        void OnNewLine(int /*anY*/) {}

        void UseAggreg(const std::complex<int> & /*aP*/,const cCumulSomIm & /*aCSI*/) {};
        void Init(const std::complex<int> & /*aP*/,int & anEl)
        {
            anEl =  mOPB.GetNext()->Som();
        }
        cArgSommeRapideIteree
        (
           const cInterfaceIm2D &   anIm1,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
        ) :
            mPVois (aSzV,aSzV),
            mArg   (anIm1),
            mOPB   (
                      mArg,
                      aP0 - mPVois,
                      aP1 + mPVois,
                      -mPVois,
                      mPVois
                    )
        {
        }
    private :
        std::complex<int>                    mPVois;
        cArgSommeRapide1Im                   mArg;
        cTplOpbBufImage<cArgSommeRapide1Im>  mOPB;
};


void TestSomIteree
     (
           cInterfaceIm2D &         anImOut,
           const cInterfaceIm2D &   anIm1,
           const std::complex<int> & aP0,
           const std::complex<int> & aP1,
           int                       aSzV
           
     );



};


#endif // _EX_OPER_ASSOC_EXTERN_H_




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
