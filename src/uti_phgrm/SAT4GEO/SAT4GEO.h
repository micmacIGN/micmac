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

[2] M. P
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

#ifndef _SAT4GEO_H_


#if ELISE_QT
    #include "general/visual_mainwindow.h"
#endif

#include "StdAfx.h"
#include <algorithm>
#include "../Apero/cCameraRPC.h"

class cSomSat; //in conjunction with cGraphHomSat to calculate the pairs
class cGraphHomSat; // class that calculates the pairs
class cAppliSat3DPipeline; // class managing the pipeline
class cCommonAppliSat3D; // class managing parameters common to all stages of the pipeline
class cAppliCreateEpi; // creates the epipolar images
class cAppliRecalRPC; // recalculates RPC for the epipolar images
class cAppliMM1P; // does per-pair matching
class cAppliFusion; //does multi-view stereo fusion


/****************************************/
/********* cCommonAppliSat3D ************/
/****************************************/

class cCommonAppliSat3D
{
    public:

        cCommonAppliSat3D();

        LArgMain &     ArgBasic();
        LArgMain &     ArgEpip();
        LArgMain &     ArgRPC();
        LArgMain &     ArgMM1P();
        LArgMain &     ArgFuse();
        std::string    ComParamPairs();
        std::string    ComParamEpip();
        std::string    ComParamRPC_Basic();
        std::string    ComParamRPC();
        std::string    ComParamMatch();
        std::string    ComParamFuse();


        cInterfChantierNameManipulateur * mICNM;


        /* Common parameters */
        bool                              mExe;
        std::string                       mDir;
        std::string                       mSH;
        bool                              mExpTxt;
		int								  mNbProc;

        /* Pairs param */
        std::string mFilePairs;
        std::string mFPairsDirMEC;
		Pt2dr       mBtoHLim;

        /* Epip param */
        bool                mDoIm;
        bool                mDegreEpi;
        Pt2dr               mDir1;
        Pt2dr               mDir2;
        int                 mNbZ;
        int                 mNbZRand;
        Pt2dr               mIntZ;
        int                 mNbXY;
        Pt2di               mNbCalcDir;
        std::vector<double> mExpCurve;
        std::vector<double> mOhP;
	bool                mXCorrecHom;
	bool                mXCorrecOri;
	bool                mXCorrecL2;

        /* Convert orientation => à verifier */
        // images and Appuis generés par CreateEpip, mOutRPC, Degre, ChSys
        std::string mOutRPC;
        int         mDegreRPC;
        std::string mChSys;

        /* Match param */
        int     mZoom0;
        int     mZoomF;
	double      mResolTerrain;
        Box2dr      mBoxTerrain;

        //bool    mCMS;
        bool    mDoPly;
		bool    mEZA;
		//bool    mHasVeg;
		//bool    mHasSBG;
		double  mInc;
        double  mRegul;
        //double  mDefCor;
        bool    ExpTxt;
        int     mSzW;
        //Pt2di   mSzW0;
        //bool    mCensusQ;
	bool   	      mMMVII;
	std::string   mMMVII_mode;
	std::string   mMMVII_ModePad;
	std::string   mMMVII_ImName;
	Pt2di         mMMVII_SzTile;
    int           mMMVII_NbProc;

        /* Bascule param */
        // Malt UrbanMNE to create destination frame
        // NuageBascule -> obligatory params not user
        std::string mNameEpiLOF;

        /* SMDM param */
        std::string mOutSMDM;

    private:
        LArgMain * mArgBasic;
		LArgMain * mArgEpip;
		LArgMain * mArgRPC;
		LArgMain * mArgMM1P;
        LArgMain * mArgFuse;

        cCommonAppliSat3D(const cCommonAppliSat3D&) ; // N.I.

};


/****************************************/
/********* cSomSat           ************/
/****************************************/

class cSomSat
{
     public :
       cSomSat(const cGraphHomSat & aGH,const std::string & aName,CameraRPC * aCam,Pt3dr aC) :
          mGH   (aGH),
          mName (aName),
          mCam (aCam),
          mC   (aC)
       {
       }

       const cGraphHomSat & mGH;
       std::string          mName;
       CameraRPC *          mCam;
       Pt3dr                mC;



       bool HasInter(const cSomSat & aS2) const;

};


/****************************************/
/*********** cGraphHomSat  **************/
/****************************************/

class cGraphHomSat
{
    public :

        friend class cSomSat;

        cGraphHomSat(int argc,char** argv);
        void DoAll();

    private :

		double CalcBtoH(const CameraRPC * , const CameraRPC * );

        std::string mDir;
        std::string mPat;
        std::string mOri;
        cInterfChantierNameManipulateur * mICNM;

        std::string mOut;

        std::list<std::string>  mLFile;
        std::vector<cSomSat *>    mVC;
        int                    mNbSom;
        double                 mAltiSol;
	    Pt2dr                  mBtoHLim;

};


/****************************************/
/********* cAppliCreateEpi **************/
/****************************************/

class cAppliCreateEpi : cCommonAppliSat3D
{
    public:
        cAppliCreateEpi(int argc, char** argv);

    private:
        cCommonAppliSat3D mCAS3D;
        std::string       mFilePairs;
        std::string       mOri;

};


/****************************************/
/********* cAppliRecalRPC  **************/
/****************************************/

class cAppliRecalRPC : cCommonAppliSat3D
{
    public:
        cAppliRecalRPC(int argc,char ** argv);

    private:
        cCommonAppliSat3D mCAS3D;
        std::string       mOri;
};



/****************************************/
/********* cAppliMM1P      **************/
/****************************************/

class cAppliMM1P : cCommonAppliSat3D
{
    public:
        cAppliMM1P(int argc, char** argv);

    private:
        cCommonAppliSat3D mCAS3D;
        std::string       mFilePairs;
        std::string       mOri;

};


/****************************************/
/********* cAppliFusion    **************/
/****************************************/

class cAppliFusion
{
    public:
        cAppliFusion(int argc,char ** argv);

        void DoAll();

        cCommonAppliSat3D mCAS3D;

    private:
		std::string AddFilePostFix();
        std::string PxZName(const std::string & aInPx);
        std::string NuageZName(const std::string & aInNuageProf);
		std::string MaskZName(const std::string & aInMask);

        std::string mFilePairs;
        std::string mOri;


};



/*******************************************/
/********* cAppliSat3DPipeline  ************/
/*******************************************/


class cAppliSat3DPipeline : cCommonAppliSat3D
{
    public:
        cAppliSat3DPipeline(int argc, char** argv);

        void DoAll();

    private:
        void StdCom(const std::string & aCom,const std::string & aPost="");

        cCommonAppliSat3D mCAS3D;

        std::string mPat;
        std::string mOri;

        bool        mDebug;

        ElTimer     mChrono;
};

#endif //_SAT4GEO_H_


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
