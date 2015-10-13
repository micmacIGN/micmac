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

#ifndef __CCAMERARPC_H__
#define __CCAMERARPC_H__

#include "StdAfx.h"
#include "../../SateLib/RPC.h"
#include "../../../include/XML_GEN/SuperposImage.h"


class CameraRPC : public cBasicGeomCap3D
{
	public:
		CameraRPC(const std::string &aNameFile, 
			  const eTypeImporGenBundle &aType, 
			  std::string &aChSys, 
			  const Pt2di &aGridSz, 
			  const std::string &aMetaFile="");
		CameraRPC(const std::string &aNameFile, 
                  const eTypeImporGenBundle &aType, 
   	              const cSystemeCoord * aChSys=0,
                  const double aAltiSol=0);

		~CameraRPC();

		Pt3dr ToSysCible(const Pt3dr &) const;
		Pt3dr ToSysSource(const Pt3dr &) const;

		Pt2dr Ter2Capteur   (const Pt3dr & aP) const;
		ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const;
		bool     PIsVisibleInImage   (const Pt3dr & aP) const;

		bool  HasRoughCapteur2Terrain() const;
		Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const;

		Pt3dr ImEtZ2Terrain(const Pt2dr & aP,double aZ) const;
        Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aProf) const;
		double ResolSolOfPt(const Pt3dr &) const ;
		bool  CaptHasData(const Pt2dr &) const;

		Pt2di SzBasicCapt3D() const;

		//utm - reauires a reimplementation!!!!!!!!!!!!
		void ExpImp2Bundle(std::vector<std::vector<ElSeg3D> > 
		     aGridToExp=std::vector<std::vector<ElSeg3D> >()) const;
		//geoc  reauires a reimplementation!!!!!!!!!!!!
		void Exp2BundleInGeoc(std::vector<std::vector<ElSeg3D> > 
		     aGridToExp=std::vector<std::vector<ElSeg3D> >()) const;
        void TestDirectRPCGen();

        /* Optical centers for a user-defined grid */
		void  OpticalCenterGrid(bool aIfSave) const;
        void  OpticalCenterOfImg();
		Pt3dr OpticalCenterOfPixel(const Pt2dr & aP) const ;
        bool  HasOpticalCenterOfPixel() const;

		

        void SetProfondeur(double aP);
		double GetProfondeur() const;
        bool ProfIsDef() const;
        void SetAltiSol(double aZ);
		double GetAltiSol() const;
		bool AltisSolIsDef() const;


        const RPC * GetRPC() const;
		const std::string & GetImName() const;

		const cSystemeCoord * mChSys;
		//std::string mSysCible;//not updated and to be removed
        
    private:
		bool   mProfondeurIsDef;
		double mProfondeur;
		bool   mAltisSolIsDef;
		double mAltiSol;

		bool   mOptCentersIsDef;

		RPC                * mRPC;
		std::vector<Pt3dr> * mOpticalCenters;
		
		Pt2di        mGridSz;
        std::string  mImName; 

		ElSeg3D F2toRayonLPH(Pt3dr &aP0,Pt3dr & aP1) const;
                
                //Pt3dr Origin2TargetCS(const Pt3dr & aP);
                //Pt3dr Target2OriginCS(const Pt3dr & aP);
		
		const std::string FindUTMCS();

        void UpdateValidity3DFromPix() const;

        void AssertRPCDirInit() const;
		void AssertRPCInvInit() const;
};

cBasicGeomCap3D * CamRPCOrientGenFromFile(const std::string & aName, const eTypeImporGenBundle aType, const cSystemeCoord * aChSys);

//dimap v1 - Simplified_Location_Model
class CameraAffine : public cBasicGeomCap3D
{
    public:
            CameraAffine(std::string const &aNameFile);
            ~CameraAffine(){};

	    ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const;
            Pt2dr    Ter2Capteur   (const Pt3dr & aP) const;
            Pt2di    SzBasicCapt3D() const;
	    double ResolSolOfPt(const Pt3dr &) const;
	    bool  CaptHasData(const Pt2dr &) const;
	    bool     PIsVisibleInImage   (const Pt3dr & aP) const;

	    Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const;

	    bool     HasOpticalCenterOfPixel() const;
	    Pt3dr    OpticalCenterOfPixel(const Pt2dr & aP) const ;
	    void Diff(Pt2dr & aDx,Pt2dr & aDy,Pt2dr & aDz,const Pt2dr & aPIm,const Pt3dr & aTer);

            //print the CameraAffine parameters 
            void ShowInfo();

    private:
	    /* Affine param */
	    //direct model
	    std::vector<double> mCDir_LON;//lambda
	    std::vector<double> mCDir_LAT;//phi

	    //inverse model
	    std::vector<double> mCInv_Line;
	    std::vector<double> mCInv_Sample;

	    /* Validity zones */
	    double mLAT0, mLATn, mLON0, mLONn;
	    double mROW0, mROWn, mCOL0, mCOLn;


            Pt2di        mSz;
	    std::string  mCamNom;


};

class BundleCameraRPC : public cCapture3D
{
	public:
	    BundleCameraRPC(cCapture3D * aCam);
	    ~BundleCameraRPC(){};

	    Pt2dr    Ter2Capteur   (const Pt3dr & aP) const;
	    bool     PIsVisibleInImage   (const Pt3dr & aP) const;
	    ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const;
	    Pt2di    SzBasicCapt3D() const; 

	    bool  HasRoughCapteur2Terrain() const;
	    bool  HasPreciseCapteur2Terrain() const;
	    Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const;
	    Pt3dr PreciseCapteur2Terrain   (const Pt2dr & aP) const;

	    double ResolSolOfPt(const Pt3dr &) const ;
            double ResolSolGlob() const ;
	    bool  CaptHasData(const Pt2dr &) const;

            Pt2dr ImRef2Capteur   (const Pt2dr & aP) const;
            double ResolImRefFromCapteur() const;

	private:
	    cCapture3D * mCam;
};

#endif

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
