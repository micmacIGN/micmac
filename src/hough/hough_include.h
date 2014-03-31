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


#ifndef _ELISE_HOUGH_INCLUDE_
#define _ELISE_HOUGH_INCLUDE_

#include "general/all.h"
#include "private/all.h"
#include "im_special/hough.h"
#include "im_tpl/max_loc.h"

#include <vector>
#include <map>



/*
     Cette classe a pour vocation d'assurer des services
    refatifs au clipping d'un polygone par une bande
    verticale.
*/

class PolygoneClipBandeVert
{
     public :

 

	    void ClipPts(ElFifo<Pt2dr> &Pcliped,REAL yO,REAL y1,const ElFifo<Pt2dr> &POrig);
	    REAL SurClipPts(REAL x0,REAL x1,const ElFifo<Pt2dr> &POrig);
		// Surf de clip d'un carre dont les deux premier sommets sont p0,p1
	    REAL SquareSurClipPts(REAL x0,REAL x1,Pt2dr p0,Pt2dr p1);

		static void Bench();

     private :

       void init(REAL yO,REAL y1);
       void AddSeg(ElFifo<Pt2dr> &,Pt2dr p0,Pt2dr p1) const;

        typedef enum 
        {
             PosGauche = -1,
             PosMilieu = 0,
             PosDroite = 1
        }  tPosBande;


        inline tPosBande PosBande(const Pt2dr & pt) const;

        static inline  Pt2dr InterDrVert(REAL x,const Pt2dr & p0,const Pt2dr & p1);

        inline Pt2dr IDVGauche(const Pt2dr & p0,const Pt2dr & p1) const;
        inline Pt2dr IDVDroite(const Pt2dr & p0,const Pt2dr & p1) const;

        REAL          mX0;
        REAL          mX1;
        ElFifo<Pt2dr> mBufClip;
        ElFifo<Pt2dr> mBufSquare;

		static void Bench(Pt2dr p0,Pt2dr q0,REAL x0,REAL x1,REAL step,PolygoneClipBandeVert &);
};



// Passe par un typedef IDIOT a cause de bug sur gcc version 2.95.2 19991024  (release)
typedef CalcMaxLoc<INT,INT,ElSTDNS less<INT> > ElH_CML;


class ElHoughImplem : public ElHough
{
              //   typedef U_INT1 tPt; // typedef U_INT1 tVal;

     public :
	

       friend class ElHoughFromFile;
       static ElHoughImplem * NewOne
              (
                      Pt2di SzXY,
                      REAL  StepRho ,
                      REAL StepTeta ,
                      Mode mode,
					  REAL RabRho,
					  REAL RabTeta
              );
       
       // INT NbX() 

     protected :
        typedef  U_INT2             tElIndex;
        typedef  Im1D<tElIndex,INT> tImIndex;

       ElHoughImplem
       (
             Pt2di SzXY,
             REAL  StepRho ,
             REAL  StepTeta,  // en equivalent pixel,
			 REAL  RabRho,
			 REAL  RabTeta
       );

	   void PostInit();

	   INT NbTetaTot()  const {return NbTeta()+2*mNbRabTeta;}

	    REAL DynamicModeValues() ;
		REAL DynamicModeGradient() ;   



    // XXX_G2S, XXX_S2G  : transfomation "Grille to Space" ou "Space to Grille"


        REAL  G2S_Teta(REAL iTeta)const{ return iTeta*mStepTeta;}
		REAL   S2G_Teta(REAL aTeta) const {return mod_real(aTeta/mStepTeta,NbTeta());}
        Pt2dr G2S_XY(Pt2dr p) const {return Pt2dr(p.x-mCX,p.y-mCY);}
        Pt2dr S2G_XY(Pt2dr p) const {return Pt2dr(p.x+mCX,p.y+mCY);}

        REAL  CentG2S_Rho(REAL iRho,INT iTeta)  const
        { 
           return iRho*mDataSRho[iTeta];
        }
        REAL  CentS2G_Rho(REAL iRho,INT iTeta)  const
        { 
           return iRho/mDataSRho[iTeta];
        }

        REAL  PosG2S_Rho(REAL iRho,INT iTeta)  const
        { 
           return CentG2S_Rho(iRho+mIRhoMin,iTeta);
        }
        REAL  PosS2G_Rho(REAL iRho,INT iTeta)  const
        { 
           return CentS2G_Rho(iRho,iTeta)-mIRhoMin;
        }

		INT IndRhoSym(INT IRho) {return -IRho - 2*mIRhoMin;}


        typedef ElSTDNS pair<Pt2di,REAL>   tCel;
        typedef  ElSTDNS vector<tCel>  tLCel;

        void SetStepRho(INT iTeta,REAL8 step)
        {
             mDataSRho[iTeta] = step;
        }
        REAL StepRhoInit() const {return mStepRhoInit;}
        REAL StepRho(INT iTeta) const {return mDataSRho[iTeta];}

    private :



		void SetImageXYCur(Im2D_U_INT1);
		void VerifIsImageXY(Im2D_U_INT1);
		void SetImageRTCur(Im2D_INT4);
        virtual void clean() = 0;
        virtual void ElemPixel(tLCel &,Pt2di) =0;
		INT GetIndTeta(INT AValue,tElIndex *,INT Nb);


        void finish();
        void Transform(Im2D_INT4,Im2D_U_INT1);
        void Transform_Ang
             (
                 Im2D_INT4,
                 Im2D_U_INT1 ImMod,
                 Im2D_U_INT1 ImAng,
                 REAL        IncAng,
                 bool        AngIsGrad  // Si vrai += 90 degre
             );
         Im2D_INT4 PdsAng(Im2D_U_INT1,Im2D_U_INT1,REAL Incert,bool);


        void MakeCirc(Im2D_INT4);


        Im2D_INT4 PdsInit();
        INT  NbCel() const {return mNbCelTot;}
        void write_to_file(const std::string &) const;
        Im2D_INT4 Pds(Im2D_U_INT1);
        Seg2d Grid_Hough2Euclid(Pt2dr) const ;
        Pt2dr Grid_Euclid2Hough(Seg2d) const; 
        INT NbRabTeta() const ;
        INT NbRabRho()  const ;   


	static ElHoughImplem * SubPixellaire
               (Pt2di,REAL,REAL,REAL RabRho,REAL RabTeta);
	static ElHoughImplem * Basic
               (Pt2di,REAL,REAL,bool Adapt,bool SubPix,REAL RabRho,REAL RabTeta);

        void CalcMaxLoc
             (
                  Im2D_INT4,
                  ElSTDNS vector<Pt2di> & Pts,
                  REAL VoisRho,
                  REAL VoisTeta,
                  REAL Vmin
             );

		bool BandeConnectedVsup
             (
                  Pt2di       p1,
                  Pt2di       p2,
                  Im2D_INT4   Im,
                  INT         VInf,
                  REAL        Tol
              );                   
        void FiltrMaxLoc_BCVS
             (
                  ElSTDNS vector<Pt2di> & Pts,
                  Im2D_INT4  Im,
                  REAL  FactInf,
                  REAL  TolGeom,
                  REAL  VoisRho,
                  REAL  VoisTeta
             );

    //  Champs 
        //  Taille de la grille en XY
   
        // centre de la grille 
        REAL  mCX;
        REAL  mCY;

        // Pas de discretisation en rho-teta
        //  Taille de la grille en rho-teta

		REAL        mRabRho;
		INT         mNbRabRho;
        REAL        mStepTeta;
		INT         mNbRabTeta;
		INT         mNbTetaTot;
		REAL        mRabTeta;
        REAL        mStepRhoInit;
        Im1D_REAL8  mStepRho;
        REAL8 *     mDataSRho;

        INT   mIRhoMin;
        INT   mIRhoMax;
        INT   mNbCelTot;
        REAL  mFactPds;

        Pt2di P0Vois(){return Pt2di(mNbRabTeta,mNbRabRho);}
        Pt2di P1Vois(){return Pt2di(mNbRabTeta+NbTeta(),mNbRabRho+NbRho());}


        Im2D_INT4    mAdrElem;
        INT **       mDataAdE;
        Im2D_U_INT2  mNbElem;
        U_INT2 **    mDataNbE;

        tImIndex    mIndRho;
        tElIndex *  mDataIRho;
        tImIndex    mIndTeta;
        tElIndex *  mDataITeta;
		int         mGetTetaTrivial;
        Im1D_U_INT1 mPds;
        U_INT1 *    mDataPds;


        Im2D_INT4    mHouhAccul;
        INT4 **      mDataHA;
        Im2D_INT4    mHouhAcculInit;
        INT4 **      mDataHAInit;
        Im2D_U_INT1  mImageXY;  // Pointe sur l'image a analsyer
        U_INT1 **    mDataImXY;
        Im2D_INT4    mImageRT;  // Pointe sur l'image a analsyer
        INT **       mDataImRT;

        Im1D_U_INT1  mPdsEcTeta;
        U_INT1 *     mDataPdsEcTeta;

		Im2D_U_INT1  mMarqBCVS;     // Tampon pour BandeConnectedVsup

		ElH_CML mCML;


		REAL mDMV_Val;
		bool mDMV_IsCalc;

};




#endif //  _ELISE_HOUGH_INCLUDE_



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
