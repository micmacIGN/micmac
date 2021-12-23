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

[2] M. Pierrot-Deseilligny, "MicMac, un logiciel de mise en correspondance
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
#include "BundleGen.h"
#include "../../../include/XML_GEN/SuperposImage.h"

typedef enum
{
    eRP_None,
    eRP_Poly
} eRefinePB;

class cRPC;
class CameraRPC;



class CameraRPC : public cBasicGeomCap3D
{
	public:
		CameraRPC(const std::string &aNameFile, 
                  const double aAltiSol=0);
		CameraRPC(const std::string &aNameFile, 
                  const eTypeImporGenBundle &aType, 
   	              const cSystemeCoord * aChSys=0,
                  const double aAltiSol=0);
		~CameraRPC();


		Pt2dr Ter2Capteur   (const Pt3dr & aP) const;
		ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const;
		bool     PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage * =0) const;

		bool  HasRoughCapteur2Terrain() const;
		Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const;

		Pt3dr ImEtZ2Terrain(const Pt2dr & aP, double aZ) const;
        Pt3dr ImEtProf2Terrain(const Pt2dr & aP, double aProf) const;
		double ResolSolOfPt(const Pt3dr &) const ;
		bool  CaptHasData(const Pt2dr &) const;

		Pt2di SzBasicCapt3D() const;


        /* Optical centers for a user-defined grid */
		void  OpticalCenterGrid(bool aIfSave) const;
        void  OpticalCenterOfImg();
        void  OpticalCenterOfImg(std::vector<Pt3dr>* aOC) const;
		Pt3dr OpticalCenterOfLine(const double & aL) const ;
		Pt3dr OpticalCenterOfPixel(const Pt2dr & aP) const ;
        bool  HasOpticalCenterOfPixel() const;

		

        void SetProfondeur(double aP);
		double GetProfondeur() const;
        bool ProfIsDef() const;
        void SetAltiSol(double aZ);
        void SetAltisSolMinMax(Pt2dr);
		double GetAltiSol() const;
		double GetAltiSolInc() const;
        Pt2dr GetAltiSolMinMax() const; // MPD => const, sinon ca ne surcharge pas la methode
		bool AltisSolIsDef() const;
        bool AltisSolMinMaxIsDef() const;
        bool IsRPC() const;
        
        const std::vector<Pt2dr> &  ContourUtile();
		const cElPolygone &  EmpriseSol() const;
		const Box2dr & BoxSol() const;

        const  cRPC * GetRPC() const;
        cRPC   GetRPCCpy() const;
        int    CropRPC(const std::string &, const std::string &, const std::vector<Pt3dr>&);
        void   SetGridSz(const Pt2di & aSz);

        void   ExpImp2Bundle(std::vector<std::vector<ElSeg3D> > aGridToExp=std::vector<std::vector<ElSeg3D> >()) const;
        virtual std::string Save2XmlStdMMName(  cInterfChantierNameManipulateur * anICNM,
                                        const std::string & aOriOut,
                                        const std::string & aNameImClip,
                                        const ElAffin2D & anOrIntInit2Cur
                    ) const;


        static cBasicGeomCap3D * CamRPCOrientGenFromFile(
        const std::string & aName, 
        const eTypeImporGenBundle aType, 
        const cSystemeCoord * aChSys);
		
       // cSystemeCoord  mChSys;

    private:
		void  SetContourUtile();
		void  SetEmpriseSol();

		bool   mProfondeurIsDef;
		double mProfondeur;
		bool   mAltisSolIsDef;
        bool   mAltisSolMinMaxIsDef;
		double mAltiSol;
        Pt2dr mAltisSolMinMax;
        std::vector<Pt2dr>  mContourUtile;
		cElPolygone mEmpriseSol;
		Box2dr      mBoxSol;


		bool   mOptCentersIsDef;

		cRPC                * mRPC;
		std::vector<Pt3dr> * mOpticalCenters;
		
		Pt2di        mGridSz;
		std::string  mInputName; 

		
        ElSeg3D F2toRayonLPH(const Pt2dr &aP, const double &aZ0, const double &aZP1) const;


        void AssertRPCDirInit() const;
		void AssertRPCInvInit() const;
};

cBasicGeomCap3D * CamRPCOrientGenFromFile(const std::string & aName, const eTypeImporGenBundle aType, const cSystemeCoord * aChSys);

//dimap v1 - Simplified_Location_Model - NOT SUPPORTED FOR NOW
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
	    bool     PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage * =0) const;

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

class cRPC
{
    public:
        friend class CameraRPC;

        cRPC(const std::string &);
        cRPC(const std::string &, const eTypeImporGenBundle &, 
             const cSystemeCoord *aChSys=0);
        ~cRPC(){};

        /* Re-save in original coordinate system */
        static std::string Save2XmlStdMMName(  cInterfChantierNameManipulateur * anICNM,
                                        const std::string & aOri,
                                        const std::string & aNameImClip,
                                        const ElAffin2D & anOrIntInit2Cur,
										const std::string & aOriOut="-RecalRPC"
                    );

        /* Save non-existing RPCs in original coordinate system */
        static std::string Save2XmlStdMMName_(cRPC &, const std::string &);
        static std::string NameSave(const std::string & aDirLoc,std::string aDirName="NEW/");
        void Show();

        /* 2D<->3D projections */
        Pt3dr DirectRPC(const Pt2dr &aP, const double &aZ) const;
        Pt2dr InverseRPC(const Pt3dr &aP) const;
        void  InvToDirRPC();

        /* Noralize / denormalize */
        Pt2dr NormIm(const Pt2dr &aP, bool aDENORM=0) const;
        vector<Pt3dr> NormImAll(const vector<Pt3dr> &aP, bool aDENORM=0) const;
        Pt3dr NormGr(const Pt3dr &aP, bool aDENORM=0) const;
        vector<Pt3dr> NormGrAll(const vector<Pt3dr> &aP, bool aDENORM=0) const;
        double NormGrZ(const double &aZ, bool aDENORM=0) const;


        double GetGrC31() const;
        double GetGrC32() const;
        double GetImRow1() const;
        double GetImRow2() const;
        double GetImCol1() const;
        double GetImCol2() const;

        Pt3di GetGrid() const;


        bool IsDir() const;
        bool IsInv() const;
        bool IsMetric() const;    
      
        /* Grid creation */
        static void SetRecGrid_(const bool  &, const Pt3dr &, const Pt3dr &, Pt3di &);
        static void GenGridAbs_(const Pt3dr &aPMin, const Pt3dr &aPMax, const Pt3di &aSz, std::vector<Pt3dr> &aGrid);
        void GenGridAbs(const Pt3di &aSz, std::vector<Pt3dr> &aGrid);
        void GenGridNorm(const Pt3di &aSz, std::vector<Pt3dr> &aGrid);
        static void GenGridNorm_(const Pt2dr aRange, const Pt3di &aSz, std::vector<Pt3dr> &aGrid);

        
        static void GetGridExt(const std::vector<Pt3dr> & aGrid, 
                         Pt3dr & aExtMin,
                         Pt3dr & aExtMax,
                         Pt3dr & aSumXYZ );
        
    
    private:
        void Initialize(const std::string &,
                   const eTypeImporGenBundle &,
                   const cSystemeCoord *);
        void Initialize_(const cSystemeCoord *);

        /* Normalized 2D<->3D projections */
        Pt3dr DirectRPCN(const Pt2dr &aP, const double &aZ) const;
        Pt2dr InverseRPCN(const Pt3dr &aP) const;

        /* Reading */
        void ReadDimap(const std::string &aFile);
        void ReadXML(const std::string &aFile);
        void ReadASCII(const std::string &aFile);
        int  ReadASCIIMeta(const std::string &aMeta, const std::string &aFile);
        void ReadEUCLIDIUM(const std::string &aFile);
        void ReadScanLineSensor(const std::string &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &);
        void ReadScanLineSensor(const std::string &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &);
        void ReadEpiGrid(const std::string &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &,
                                std::vector<Pt3dr> &);

        
        /* Change coordinate system */
        void ChSysRPC(const cSystemeCoord &);
        void ChSysRPC(const cSystemeCoord &,
                       double (&aDirSNum)[20], double (&aDirLNum)[20],
                       double (&aDirSDen)[20], double (&aDirLDen)[20],
                       double (&aInvSNum)[20], double (&aInvLNum)[20],
                       double (&aInvSDen)[20], double (&aInvLDen)[20]);
        void ChSysRPC_(const cSystemeCoord &, 
                        const Pt3di &aSz, 
                        double (&aDirSNum)[20], double (&aDirLNum)[20],
                        double (&aDirSDen)[20], double (&aDirLDen)[20],
                        double (&aInvSNum)[20], double (&aInvLNum)[20],
                        double (&aInvSDen)[20], double (&aInvLDen)[20],
                        bool PRECISIONTEST=1);




        /* Fill-in a cubic polynomials */
        void CubicPolyFil(const Pt3dr &aP, double (&aPTerms)[20]) const;
        void DifCubicPolyFil(const Pt3dr &aP, double &aB, double (&aPTerms)[39]) const;
        
        //for alternative equations
        void FilBD(double (&ab)[20], double (&aU)[20], double (&aB)) const;
        void DifCubicPolyFil(double &aB, double &aDenomApprox, double(&aU)[20], double (&aPTerms)[39]) const;


        /* Learn RPCs */
        void LearnParamNEW(std::vector<Pt3dr> &aGridIn,
                         std::vector<Pt3dr> &aGridOut,
                         double (&aSol1)[20], double (&aSol2)[20],
                         double (&aSol3)[20], double (&aSol4)[20]);
        void LearnParam(std::vector<Pt3dr> &aGridIn,
                         std::vector<Pt3dr> &aGridOut,
                         double (&aSol1)[20], double (&aSol2)[20],
                         double (&aSol3)[20], double (&aSol4)[20]);
        
        void CalculRPC(     const vector<Pt3dr> &, 
                            const vector<Pt3dr> &, 
                            const vector<Pt3dr> &, 
                            const vector<Pt3dr> &, 
                            double (&aDirSNum)[20], double (&aDirLNum)[20],
                            double (&aDirSDen)[20], double (&aDirLDen)[20],
                            double (&aInvSNum)[20], double (&aInvLNum)[20],
                            double (&aInvSDen)[20], double (&aInvLDen)[20],
                            bool PRECISIONTEST=1);
    

        /* Validity utils */
        void ReconstructValidityxy();
        void ReconstructValidityXY();
        void ReconstructValidityH();
        void FillAndVerifyBord(double &aL, double &aC,
                               const Pt3dr &aP1, const Pt3dr &aP2,
                               const std::list< Pt3dr > &aP3,
                               std::vector<Pt3dr> & aG3d, std::vector<Pt3dr> & aG2d);
		void UpdateGrC(Pt3dr& );

        /* Update scales, offsets */
        void NewImOffScal(const std::vector<Pt3dr> & aGrid);
        void NewGrOffScal(const std::vector<Pt3dr> & aGrid);
        void NewGrC(double &aGrC1min, double &aGrC1max,
                     double &aGrC2min, double &aGrC2max,
                     double &aGrC3min, double &aGrC3max);
        
        

        void GetGrC1(vector<double>& aC) const;
        void GetGrC2(vector<double>& aC) const;
        void GetImOff(vector<double>& aOff) const;
        void GetImScal(vector<double>& aSca) const;
        void GetGrOff(vector<double>& aOff) const;
        void GetGrScal(vector<double>& aSca) const;




        void SetRecGrid();

        void SetPolyn(const std::string &);
        
        bool AutoDetermineRPCFile(const std::string &) const;
        bool AutoDetermineGRIDFile(const std::string &) const;

        template <typename T>
        void FilLineNumCoeff(T& , double (&)[20] ) const;
        template <typename T>
        void FilLineDenCoeff(T& , double (&)[20] ) const;
        template <typename T>
        void FilSampNumCoeff(T& , double (&)[20] ) const;
        template <typename T>
        void FilSampDenCoeff(T& , double (&)[20] ) const;

        void UpdateRPC(double (&aDirSNum)[20], double (&aDirLNum)[20],
                       double (&aDirSDen)[20], double (&aDirLDen)[20],
                       double (&aInvSNum)[20], double (&aInvLNum)[20],
                       double (&aInvSDen)[20], double (&aInvLDen)[20]);

	void SetType(const eTypeImporGenBundle& aT) { mType=aT; };

        bool ISDIR;
        bool ISINV;
        bool ISMETER;

        /* Direct projection RPC coefficients */
        double mDirSNum[20], mDirSDen[20], mDirLNum[20], mDirLDen[20];
        /* Inverse projection RPC coefficients */
        double mInvSNum[20], mInvSDen[20], mInvLNum[20], mInvLDen[20];

        /* Use the correcting polyn if available */
        cPolynomial_BGC3M2D * mPol;
        eRefinePB mRefine;
        
        /* Coordinate system change */
        cSystemeCoord mChSys;

        /* Validity zone: offsets, scales, coordinate extents */
        double mImOff[2], mImScal[2];
        double mGrOff[3], mGrScal[3];

        double mImRows[2], mImCols[2];
        double mGrC1[2], mGrC2[2], mGrC3[2];

        /* Grid to recompute RPCs */
        Pt3di mRecGrid;

        std::string mName;

	/* RPC type */
        eTypeImporGenBundle mType;
};

class cRPCVerf
{
    public:
        cRPCVerf(const CameraRPC &aCam, const Pt3di &aSz);
        
        void Do(const std::vector<Pt3dr> &aGrid=std::vector<Pt3dr>());
        void Compare2D(std::vector<Pt2dr> &aGrid2d) const;
        void Compare3D(std::vector<Pt3dr> &aGrid3d) const;

        std::vector<Pt3dr> mGrid3dFP; //forward projected grid
        std::vector<Pt2dr> mGrid2dBP;  //backward projected grid
    
    private:
        const Pt3di mSz;
        const CameraRPC * mCam;
        std::vector<Pt3dr> mGrid3d;   //input grid unnormalized

};

#endif

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un pro
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
