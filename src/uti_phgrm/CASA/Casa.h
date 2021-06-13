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


#ifndef _ELISE_CASA_ALL_H_
#define _ELISE_CASA_ALL_H_


#include "cParamCasa.h"



// Dans un faceton la normale est orientee dans le meme sens que les rayon camera
class cFaceton
{
     public :
        cFaceton(double aPds,Pt2dr anIndex,Pt3dr aCdg,Pt3dr aNorm);
        cFaceton();  // Renvoie un jerk

        const Pt2dr & Index() const;
        const Pt3dr & Centre() const;
        const Pt3dr & Normale() const;
        ElSeg3D DroiteNormale() const;

        // Renvoie le faceton en position moyenne (au sens du min
        // des ecarts quad a l'ensemble
        static int  GetIndMoyen(const std::vector<cFaceton> & aVF);

     // Terminologie claire pour un cylindre, de maniere generale, est ce que la normale
     // est dans le sens des z (de la coord anam)  decroissant 
         bool IsFaceExterne(const cInterfSurfaceAnalytique &) const;
         bool Ok() const;

     private :

         double mPds;
         Pt2dr  mIndex;
         Pt3dr  mCentre;
         Pt3dr  mNormale;
         bool mOk;
};

// On accumule l'info dans le moment d'inertie. Le faceton
// compile, est defini par son centre et sa normale (calcule
// par diag de la mat d'inertie).  L'index est un point
// moyen de l'image

class cAccumFaceton
{
    public :
       void Add(const Pt2dr & anIndex,const Pt3dr  & ,double aPds);
      cAccumFaceton();
      cFaceton   CompileF(const cElNuage3DMaille &);
    private  :
       
        double mSomPds;
        Pt3dr  mSomPt;
        Pt2dr  mSomInd;
        ElMatrix<double> mMoment;
};


struct cResEcartCompense
{
    public :
       double  mMoyQuad;
       double  mMoyHaut;
};


struct   cOneSurf_Casa
{
    public :
        cOneSurf_Casa();
        void ActiveContrainte(cSetEqFormelles & aSet);
        void  Compense(const cCasaEtapeCompensation & anEtape,bool First);

         bool IsFaceExterne(const cInterfSurfaceAnalytique &,double aTol) const;

        std::vector<cFaceton>             mVF;
        cInterfSurfAn_Formelle *          mISAF;
        double                            mBestScore;
        Video_Win *                       mW;
        cFaceton *                        mFMoy;
        std::string                       mName;
        cResEcartCompense                 mREC;
};


class cAppli_Casa
{
    public :
       cAppli_Casa( cResultSubstAndStdGetFile<cParamCasa> aParam);

    private :
         void AddNuage2Surf
              (
                 const cSectionLoadNuage&,
                 const cNuageByImage &,
                 cOneSurf_Casa& aSurf
              );
         cOneSurf_Casa * InitNuage(const cSectionLoadNuage &);

         void EstimSurf ( cOneSurf_Casa & aSurf,
                          const cSectionEstimSurf & aSES
                        );

         void Compense(const cCasaSectionCompensation &);
         void OneEtapeCompense(const cCasaEtapeCompensation &);

         const cInterfSurfaceAnalytique * UsePts(const cInterfSurfaceAnalytique *);
         void UsePtsCyl();


         void EstimeCylindreRevolution
              (
                  cOneSurf_Casa & aSurf,
                  const cSectionEstimSurf & aSES
              );
         void TestCylindreRevolution
              (
                 cOneSurf_Casa & aSurf,
                 const cFaceton & aFc1,
                 const cFaceton & aFc2
              );



         cParamCasa &                      mParam;
         cInterfChantierNameManipulateur * mICNM;
         std::string                       mDC;

         std::vector<cOneSurf_Casa *>       mVSC;
         cSetEqFormelles                   mSetEq;



         cInterfSurfaceAnalytique *       mSAN;
        // Peut prendre une des formes suivantes
        cCylindreRevolution *             mBestCyl;
};




#endif //  _ELISE_CASA_ALL_H_




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
