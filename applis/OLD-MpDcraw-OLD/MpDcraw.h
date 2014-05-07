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
// #include "anag_all.h"

#include "general/all.h"
#include "private/all.h"

#include "im_tpl/image.h"
#include "XML_GEN/all.h"

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

namespace NS_MpDcraw
{

class cArgMpDCRaw;
class cOneChanel;
class cNChannel;


   //------------------------------

class cArgMpDCRaw : public cAppliBatch
{
    public :
       cArgMpDCRaw(int argc,char ** argv);

       Fonc_Num  FlatField(const cMetaDataPhoto & aMDP,const std::string & aNameFile);

       L_Arg_Opt_Tiff  ArgMTD() const;
       void Exec();
       void DevJpg();

       bool Cons16B() const;
       bool Adapt8B() const;
       double Dyn() const;
       bool IsToSplit(const std::string &) const;
       bool GrayBasic() const;
       bool UseFF() const;
       bool Diag()   const;
       bool ColBasic() const;
       bool GrayReech() const;
       bool ColReech() const;
       const std::vector<double> &  ClB() const;
       const std::string &  NameCLB() const;
       const std::string &  MastCh() const;
       const double &  ScaleMast() const;
       cBayerCalibGeom *  BayerCalib() const;
       cInterpolateurIm2D<REAL4> * Interpol() const;


       const std::vector<double>  &  WB() const;  // White Balance
       bool  WBSpec () const;
       const std::vector<double>  &  PG() const;  // White Balance
       bool  PGSpec () const;
       bool  NameOriIsPrefix() const;
       bool   Add16_8B() const;
       double Gamma() const;  // A priori Gamma applicable sur le gris
       double EpsLog() const;  // A priori Gamma applicable sur le gris
       const std::string &  CamDist() const;
       const std::string &  HomolRedr()   const;
       const double &  ExpTimeRef() const;
       const double &  IsoSpeedRef() const;
       const double &  DiaphRef() const;
       const std::string & Extension() const;
       const std::string & StdExtensionAbs() const;
       const double & Offset() const;

       const bool  & DoSplit() const;
       const int   & SpliMargeInt() const;
       const Pt2di & SpliMargeExt() const;

       bool  SwapRB(bool aDef) const;
       const bool & ConsCol() const;

       std::string NameRes(const std::string & aNameFile,const std::string & aName,const std::string & Pref="") const;


    private  :
       int                       mCons16Bits;
       int                       m8BitAdapt;
       double                    mDyn;
       double                    mGamma;
       double                    mEpsLog;
       std::string               mSplit;
       bool                      mGB;
       bool                      mCB;
       bool                      mGR;
       bool                      mCR;
       bool                      mDiag;                           
       bool                      mConsCol;                           
       std::vector<double>       mClB;
       std::string               mNameCLB;
       std::string               mMastCh;  // Cannal maitre
       double                    mScaleMast;  // En cas de reech, taille du maitre
                                              // Typiquement 2.0
       cBayerCalibGeom *             mBayerCalibGeom;
       std::string                   mCal;
       cInterpolateurIm2D<REAL4> *   mInterp;
       double                        mBicubParam;
       int                           mSzSinCard;

       std::vector<double>           mWB;  // White Balance
       bool                          mWBSpec;
       std::vector<double>           mPG;  // White Balance
       bool                          mPGSpec;
       int                           mNameOriIsPrefix;
       int                           mAdd16_8B;
       std::string                   mCamDist;
       std::string                   mHomolRedr;
       double                        mExpTimeRef;
       double                        mDiaphRef;
       double                        mIsoSpeedRef;
       std::string                   mImRef;
       std::string                   mExtension;
       std::string                   mExtensionAbs;
       double                        mOfs;
     
       bool                          mDoSplit;
       int                           mSplitMargeInt;
       Pt2di                         mSplitMargeExt;

       int                           mSwapRB;
       std::string                   mNameOutSpec;
       bool                          mUseFF;

};

class cOneChanel  : public ElDistortion22_Gen
{
    public :
        cOneChanel
	(
	    const cNChannel *,
	    const std::string &,
	    Im2D_REAL4 aFullImage,
	    Pt2di      aP0,
	    Pt2di      aPer // Typiquement 2,2
        );
	void SauvInit();
	// Le masque a Full resolution
	Fonc_Num MasqChannel() const;
	const std::string & Name() const;

	void InitParamGeom
	     (
	          const std::string& aMastCh,
		  double             aScale,
	          cBayerCalibGeom *,
		  cInterpolateurIm2D<REAL4> *
	     );
	void MakeInitImReech();

       Pt2dr   ToRef(const Pt2dr &) const;
       Pt2dr   FromRef(const Pt2dr &) const;
       Im2D_REAL4      ImReech();

       const Pt2di & P0() const;
       const Pt2di & Per() const;
    private :
       const cNChannel *    mNC;
       std::string          mName;
       Pt2di                mSz;
       Pt2di                mP0;
       Pt2di                mPer;
       Im2D_REAL4           mIm;
       TIm2D<REAL4,REAL8>   mTIm;

       Im2D_REAL4           mImReech;
       TIm2D<REAL4,REAL8>   mTIR;
       Im2D_Bits<1>         mIMasq;
       TIm2DBits<1>         mTIM;
       Pt2di                mSzR;
       cDbleGrid *                  mGridColM2This;
       cDbleGrid *                  mGridToFusionSpace;
       cInterpolateurIm2D<REAL4> *  mInterp;
       bool                         mIsMaster;
       double                       mScale;
       CamStenope *                 mCamCorDist;
       cElHomographie *             mHomograRedr;
       cElHomographie *             mInvHomograRedr;

       Pt2dr Direct(Pt2dr aP) const;       // To Ref
       bool OwnInverse(Pt2dr & aP) const;  // From Ref
       

};


class cNChannel
{
    public :
        void  Split(const cArgMpDCRaw & anArg,const std::string & aPost,Tiff_Im aFileIn);

        // Tres sale, provisoire , car met en dur la structure RGGB
        static cNChannel Std(const cArgMpDCRaw & ,const std::string &);
	void SauvInit();
	const cArgMpDCRaw & Arg() const;

	std::string NameRes(const std::string & aName,const std::string & Pref="") const;

	GenIm::type_el  TypeOut(bool Signed) const;



	cOneChanel &    ChannelFromName(const std::string &);

        void MakeImageDiag(Im2D_REAL4 aFulI,const std::string & Ch1,const std::string & Ch2,const cArgMpDCRaw & anArg);

        Pt2di I2Diag(const Pt2di & aP) const;
        Pt2di Diag2I(const Pt2di & aP) const;

        CamStenope * CamCorDist() const;
        cElHomographie *   HomograRedr() const;
        cElHomographie *   InvHomograRedr() const;


    private :
      cNChannel
      (
	  const cArgMpDCRaw &  anArg,
          const std::string &  aNameFileInit,
          Im2D_REAL4 aFullImage,
          int        aNbC,
	  const char **    Names,
          Pt2di*     mVP0,
	  Pt2di      aPer
      );

      const cArgMpDCRaw &           mArg;
      std::string                   mFullNameFile;
      std::string                   mNameDir;
      std::string                   mNameFile;
      std::string                   mNameFileInit;
      std::vector<cOneChanel>       mVCh;

      Pt2di                         mSzR;

      // Gestion des images diag
      Pt2di mP0Diag;
      Pt2di mOfsI2Diag;
      int   mSDiag;
      Pt2di mSzDiag;
      CamStenope *                   mCamCorDist;
      cElHomographie *               mHomograRedr;
      cElHomographie *               mInvHomograRedr;

};

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
