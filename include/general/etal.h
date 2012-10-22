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
/*
    bin/EPExePointeInitPolyg  param.txt NumCible  : pour saisir

Script : 

    EPExeRechCibleInit  param.txt   :  recherche des cibles avec modeles a priori 
                                      (sur les images ListeImagesInit)

    EPExeCalibInit      param.txt   :  modele radiale initial

    EPExeRechCibleDRad param.txt   : recherche des cibles sur toutes les
                                     images avec un premier modele radiale
 
    EPExeCalibFinale  param.txt :  modele radial final.


    bin/EPtest param.txt aKindIn aKindOut  [XML=]

             aKindIn = NoGrid |  NonDeGrille.xml
             aKindOut = Hom | NoDist | Pol3 | ... | Pol7 | DRad | PhgrStd
    
*/


/*
    Salut le LOEMI

    A Faire :

     - ecriture de Sz dans param_calcule
     - Le Type "lazy Tiff"


     :
*/


typedef enum
{
    eUCI_Jamais,
    eUCI_Toujours,
    eUCI_Only,
    eUCI_OnEchec
} eUseCibleInit;

class cParamRechCible
{
     public :
        cParamRechCible(bool StepInit,INT SzW,REAL DistConf,REAL EllipseConf);

	bool  mStepInit;
	INT   mSzW;
	REAL  mDistConf;
	REAL  mEllipseConf;
	eUseCibleInit  mUseCI;
};

class cParamEtal : public cComplParamEtalPoly
{
	public :
	   cParamEtal(int,char **);
	   static cParamEtal FromStr(const std::string & aName);
	   cParamEtal ParamRatt();  // Si a un nom de ratt

	   INT Zoom() const;
	   const std::string & Directory() const;
	   const std::string & NameFile() const;
	   const std::string & PrefixeTiff() const;
	   const std::string & PostfixeTiff() const;
	   REAL   FocaleInit() const;
	   Pt2di  SzIm() const;

	   std::string NameTiff(const std::string Im) const;
	   std::string NameTiffSsDir(const std::string Im) const;
	   Tiff_Im    TiffFile(const std::string Im) const;

	   const cParamRechCible & ParamRechInit() const;
	   const cParamRechCible & ParamRechDRad() const;

	   const std::vector<std::string>  & ImagesInit() const;
	   const std::vector<std::string>  & AllImagesCibles() const;
	   REAL SeuilCorrel() const;

	   std::string NameCible3DPolygone() const;
	   std::string NameImPolygone() const;
	   std::string NamePointePolygone() const;
	
           const std::string & NameFileRatt() const;
	   bool HasFileRat() const;
           const std::string & NamePosRatt() const;
	   const std::vector<std::string>& CamRattachees() const;

	   const std::string & NameCamera() const;

	   INT   EgelsConvImage();
	   Pt3dr CibDirU() const;
	   Pt3dr CibDirV() const;
	   INT  CibleDeTest ()   const;
	   const std::string  &  ImDeTest() const;
	   bool  AcceptCible(INT) const;
	   const std::vector<INT>  &  CiblesRejetees() const;
	   bool CDistLibre(bool Def) const;
	   INT  DegDist() const;
	   bool MakeImagesCibles() const;
           REAL SeuilCoupure () const;

           bool InvYPointe() const;

           std::string ANameFileExistant();
           Pt2di RabExportGrid() const;
	   bool CalibSpecifLoemi() const;
	   cNameSpaceEqF::eTypeSysResol  TypeSysResolve() const;

	   std::string  NamePolygGenImageSaisie () const;
	   std::string  ShortNameImageGenImageSaisie () const;
	   std::string  FullNameImageGenImageSaisie () const;
	   double               ZoomGenImageSaisie () const;
	   bool                 HasGenImageSaisie () const;

	   double StepGridXML() const;
	   bool   XMLAutonome() const;
	   int    NbPMaxOrient() const;
	   double TaillePixelExportLeica() const;
	   bool   ModeMTD() const;
	   bool   ModeC2M() const;

	   const std::string & PointeInitIm();
	   double   DefLarg() const;  
           bool CalledByItsef() const;
           int  ByProcess() const;
           bool DoSift() const;
           bool DoSift(const std::string & aName) const;

           const std::string  & KeyExportOri() const;
           const std::string  & KeySetOri() const;
           const int   & DoGrid() const;

           const std::string & PatternGlob() const;
           double  SeuilRejetEcart() const;
           double  SeuilRejetAbs() const;
           double  SeuilPonder() const;

           std::string  mOrderComp;

	private :
	   void FilterImage(std::vector<std::string> &  mVIM);
	   void InitFromFile(const std::string & aName);

           std::vector<std::string>  mImages;
	   std::vector<std::string>  mImagesInit;
	   std::vector<std::string>  mCamRattachees;


	   std::string               mNameCamera;
           std::string               mNameFileRatt;
           std::string               mNamePosRattach;
	   std::string               mPolygoneDirectory;
	   std::string               mNameCible3DPolygone;
	   std::string               mNameImPolygone;
	   std::string               mNamePointePolygone;

	   std::string               mDirectory;
	   std::string               mNameFile;
	   std::string               mPrefixe;
	   std::string               mPostfixe;
	   REAL                      mFocaleInit;
	   INT                       mZoom;
	   Pt2di                     mSzIm;
	   cParamRechCible           mPRCInit;
	   cParamRechCible           mPRCDrad;
	   REAL                      mSeuilCorrel;
	   INT                       mEgelsConvImage;
	   Pt3dr                     mCibDirU;
	   Pt3dr                     mCibDirV;
	   INT                       mCibleDeTest;
	   std::string               mImDeTest;
	   std::vector<INT>          mCiblesRejetees;
	   INT                       mCDistLibre;
	   INT                       mDegDist;
	   INT                       mMakeImageCibles;
           REAL                      mSeuilCoupure;
           INT                       mInvYPointe;
           INT                       mRabExportGrid;
	   // Par defaut calcule sur le nom de la premiere image
	   
	   cNameSpaceEqF::eTypeSysResol  mTypeSysResolve;

	   std::string              mNamePolygGenImageSaisie;
	   std::string              mNameImageGenImageSaisie;
	   double                   mZoomGenImageSaisie;
	   double                   mStepGridXML;
	   int                      mXMLAutonome;
	   int                      mNbPMaxOrient;
	   double                   mTaillePixelExportLeica;

	   int                      mModeC2M;

           int                      mByProcess;
           int                      mCalledByItsef;

	   std::string              mPointeInitIm;
// 1/2 largeur du pixel pour la convol,  0.5 = capteur "parfait"
	   double                   mDefLarg;
           cElRegex *               mAutomSift;
           std::string              mKeyExportOri;
           std::string              mKeySetOri;
	   int                      mCalibSpecifLoemi;
	   int                      mDoGrid;

           std::string mPatternGlob;

           double  mSeuilRejetEcart;
           double  mSeuilRejetAbs;
           double  mSeuilPonder;
};


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
