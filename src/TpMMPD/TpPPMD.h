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
#ifndef __TD_CAMERA__
#define __TD_CAMERA__

#include "StdAfx.h"

/* ===========================================

   Ce fichier contient la definition de quelques classe donnant acces au fonctionallité
   MicMac de la manière la plus simple possible.

   Pour quelque classe simples, on se contente de rappeler l'existance des classe MicMac

==================================================*/


//     Pt2dr  classe definissant les points 2D
//     Pt3dr  classe definissant les points 3D
//

class cTD_Prof;

class cTD_Camera
{
     public :
        
        // Constructeur a partir d'un fichier XML , peut etre une calibration interne ou externe
        cTD_Camera(const std::string &);

        // Sauvegarde dans un fichier
        void Save(const std::string &) const;

        // Fonction de projection  Terrain  -> Image
        Pt2dr Ter2Image(const Pt3dr &) const;

        // Relevement dans l'espace
        std::vector<cTD_Camera> RelvtEspace
                                (
                                    const Pt3dr & aPTer1, const Pt2dr & aPIm1,
                                    const Pt3dr & aPTer2, const Pt2dr & aPIm2,
                                    const Pt3dr & aPTer3, const Pt2dr & aPIm3
                                );

        // Focale et coeff en R3

            double  Focale () const;
            double  R3 () const;
            Pt2dr   SzCam() const ;

            cTD_Camera NewCam(double aFoc , double aR3);
            

        // Manipulation géométrique 

           Pt2dr Ter2Im(const Pt3dr & aPTer) const;
           Pt3dr ImAndProf2Ter(const Pt2dr & aPTer,double aProf) const;
           double ProfMoy() const;
           double StepProfOnePixel(const cTD_Camera &) const;
          
     private :
        friend class cTD_Prof;

        std::string                mName;
        cCalibrationInternConique  mCIO;
        CamStenope *               mCS;
};

int TD_EntierAleatoire(int aN);  // Renvoie un entier au hasrd entre 1 et N

class cTD_SetAppuis
{
     public :
        cTD_SetAppuis(const std::string &);

        const std::vector<Pt3dr> & PTer() const {return mPTer;}
        const std::vector<Pt2dr> & PIm() const {return mPIm;}

     private :
         std::vector<Pt3dr> mPTer;
         std::vector<Pt2dr> mPIm;
};


class cTD_Im
{
     public :
        cTD_Im (int anX,int anY);
        static cTD_Im  FromString(const std::string &);
        cTD_Im  ImageMoy(int aSzW,int aNbIter);
        cTD_Im  ImageReduite(double aFact);



        void Save(const std::string &);

        float GetVal(int anX,int anY) const {return mTIm.get(Pt2di(anX,anY));}
        bool Ok(int anX,int anY) const  {return mTIm.inside(Pt2di(anX,anY));}
        Pt2di Sz() const ;

        void  SetVal(int anX,int anY,float aVal) {return mTIm.oset(Pt2di(anX,anY),aVal);}

     private :
        Im2D<float,double>   mIm;
        Pt2di                mSz;
        TIm2D<float,double>  mTIm;
};


#endif // __TD_CAMERA__

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
