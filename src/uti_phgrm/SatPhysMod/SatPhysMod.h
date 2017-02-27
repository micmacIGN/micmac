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

#ifndef _SAT_PHYS_MOD_H_
#define _SAT_PHYS_MOD_H_

#include "StdAfx.h"

// MPD => me demande si cette librairie ne devrait pas plutot faire  partie d'Apero ?
#include "../../uti_phgrm/Apero/cCameraRPC.h"

//  cPushB_GeomLine :  Class for the geometry of one line
class cPushB_GeomLine;

//  cPushB_PhysMod :  Absract class for push broom physical model
//   a cPushB_PhysMod contains many cPushB_GeomLine
class cPushB_PhysMod;


// cRPC_PushB_PhysMod : specilization of cPushB_PhysMod to RPC
class cRPC_PushB_PhysMod;

//================ SEUILS ==============

typedef enum  
{
    eMRP_None,    
    eMRP_Direct,    
    eMRP_Invert  
} eModeRefinePB;

class cPushB_GeomLine
{
    public :
        cPushB_GeomLine(const cPushB_PhysMod * aPBPM,int aNbSampleX,double anY);
        const Pt3dr & Center() const;
        const Pt3dr & CUnRot() const;
        Pt3dr & CUnRot() ;

        const double  & MoyResiduCenter() const;
        const double  & MaxResiduCenter() const;
        const double  & MoyDistPlan() const;
        const double  & MaxDistPlan() const;

        Pt3dr  DirOnPlan(const Pt3dr & aP) const;
        Pt2dr  PIm(int aKx) const;
        double XIm(int aKx) const;
        const std::vector<double> Calib() const;
        ElMatrix<double>  MatC2M() const;
        ElMatrix<double>  MatC1ToC2(cPushB_GeomLine & aCam2) const;


    private :

        const cPushB_PhysMod *  mPBPM;
        Pt3dr                   mCenter;
        Pt3dr                   mCUnRot;
        int                     mNbX;
        double                  mY;
        double                  mResInt;     // Average Residual of intersection
        double                  mMaxResInt;  // Max Residual of intersection
        double                  mMoyDistPlan;
        double                  mMaxDistPlan;
        cElPlan3D               mPlanRay;
        Pt3dr                   mAxeX;  // satellite displacement 
        Pt3dr                   mAxeY;  // sensor aligned
        Pt3dr                   mAxeZ;  // Point to vertical, to the earth
        std::vector<double>     mCalib;  // Inside the plane the vector (1,mCalib[aK]) is the Dir
        // std::vector<Pt3dr>      mDirs;
};

class cPushB_PhysMod
{
    public :
        Pt2di Sz() const;
        double CoherAR(const Pt2dr & aPIm) const;
        void CoherARGlob(int aNbPts,double & aMoy,double & aMax) const;
        // Value to use, one or none can be refined
        Pt2dr   GeoC2Im(const Pt3dr & aP)   const ; // Invert
        ElSeg3D Im2GeoC(const Pt2dr & aP)   const ;  // Direct

        void ShowLinesPB(bool Det);

        // Basic speed in m/pixel
        Pt3dr   Rough_GroundSpeed(double anY) const;
        Pt3dr   RoughPtIm2GeoC_Init(const Pt2dr & aPIm) const; // Midlle of Im2GeoC_Init


       // Initial value, not refined

        virtual  ElSeg3D Im2GeoC_Init (const Pt2dr & aPIm) const = 0;  // En Geo Centrique
        virtual  Pt2dr   GeoC2Im_Init   (const Pt3dr & aP)   const = 0;  // En Geo Centrique

        bool    SwapXY() const;

    protected :

        cPushB_PhysMod(const Pt2di & aSz,eModeRefinePB,const Pt2di & aSzGeoL);
        void PostInit() ; // Initialisation that must be done afterward because of virtual things
        void PostInitLinesPB() ; // Initialisation that must be done afterward because of virtual things


        static const double TheEpsilonRefine;


    // Suppose the the Direct function Im2GeoC_Init is correct, and compute the invert so to have a "perfect" coherence
        Pt2dr   GeoCToIm_Refined   (const Pt3dr & aP)   const ;
    // Suppose the the Invert function GeoC2Im_Init is correct, and compute the Direct so to have a "perfect" coherence
        ElSeg3D Im2GeoC_Refined   (const Pt2dr & aP)   const ;
        Pt3dr   Im2GeoC_Refined(const Pt2dr & aPIm,Pt3dr aP0Ter,const Pt3dr &aU,const Pt3dr & aV) const;

        Pt2di mSz;
        Pt2di mSzGeoL;
        eModeRefinePB       mModeRefine;
        std::vector<cPushB_GeomLine *>  mLinesPB;
        double                          mMoyRay;
        double                          mMoyAlt;
 // Indicateur des ecarts entre la modelisation physique et le RPC ou autre; max et moyen
        double                          mMoyRes;    // Residu d'intersection des lignes
        double                          mMaxRes;
        double                          mMoyPlan;   // Residu des directions d'un plan
        double                          mMaxPlan;
        double                          mMoyCalib;  // Residu par rapport a la calib moy
        double                          mMaxCalib;


        double                          mPeriod;
        double                          mDureeAcq;
        std::vector<double>             mCalib;
        bool                            mSwapXY;
};

class cRPC_PushB_PhysMod : public cPushB_PhysMod
{
    public :
        static cRPC_PushB_PhysMod * NewRPC_PBP(const cRPC & aRPC,eModeRefinePB,const Pt2di &aSzGeoL);
 //  LatLon  <-> Image
        Pt2dr RPC_LlZ2Im(const Pt3dr & aLlZ) const;
        Pt3dr RPC_ImAndZ2LlZ(const Pt2dr & aPIm,const double & aZ) const;
 //  Geo Centrique <-> Image
        ElSeg3D Im2GeoC_Init(const Pt2dr & aPIm) const; 

        Pt2dr   GeoC2Im_Init   (const Pt3dr & aP)   const;
    private :

        
        cRPC_PushB_PhysMod(const cRPC & aRPC,eModeRefinePB,const Pt2di & aSzGeoL);


        static const double ThePdsRay;
        static const double TheMinDeltaZ;

        cRPC  mRPC;
        cSysCoord * mWGS84Degr;
        double      mZ0Ray;  // Z used for Ray computation
        double      mZ1Ray;  // Z used for Ray computation

};





#endif // _SAT_PHYS_MOD_H_

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
