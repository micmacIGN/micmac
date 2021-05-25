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

#include "general/all.h"
#include "private/all.h"
#include "im_tpl/image.h"
#include "XML_GEN/all.h"
#include <algorithm>

using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;

class cCC_Appli;
class cCC_OneChannel;
class cCC_OneChannelSec; //  : public cCC_OneChannel
class cChannelIn;


#define aRabResize    300
#define aRabResizeOut 5
#define aRabInput  10
#define aRabOutput  10

#define DEBUG 0
/*
  La classe representant un fichier d'entree est cCC_OneChannel

  Une  cCC_OneChannel contient plusieurs

           cChannelIn

   Typiquement, il y a en aura 1 ou 3 (cas RGB).

   Le ChannelCmpCol indique pour chacun des canaux vers ou il
   pointe.

*/


class cChannelIn
{
   public :
      cChannelIn
      (
          cCC_Appli & ,
          GenIm::type_el aTypeIn,
	  cCC_OneChannel &,
          Pt2di aSz,
	  const cChannelCmpCol &
      );
      void Resize(const Pt2di &);
      Output OutForInit();
      const cChannelCmpCol & CCC();
      double      Pds() const;
      double  GetInterpolPFus(const Pt2dr & aPFus,bool & Ok);

      Pt2dr PFus2PIm(const Pt2dr & aPFus) const; 
      virtual ~cChannelIn();


   private :
      cCC_Appli &                   mAppli;
      cCC_OneChannel &              mCOC;
      const cChannelCmpCol &        mCCC;
      double                        mPds;
      double                        mDyn;
      double                        mOffset;

      Im2DGen *                     mIm;
      cIm2DInter *                  mImI;
      Pt2di                         mP0K;
      Pt2di                         mP1K;
};


//  En tant que distortion, Direct va de l'espace image brute
//  vers l'espace de fusion. Avant toute homotetie translation
//  due au clip.
//
//
//  Pour l'image maitresse , directe vaut soit
//    *  Id (cas non corrigee)
//    * DistInverse, de la camera, donc (en mode M->C) Direct
//       de la distorsion
//
//
//
//



class cCC_OneChannel  : protected ElDistortion22_Gen
{
   public :
       friend class  cOC_Dist;

       cCC_OneChannel
       (
            double   aScale,
            const cImageCmpCol &,
	    cCC_Appli &,
	    cCC_OneChannel * Master=0
       );
       CamStenope & Cam();



        // Memorise le decalage out,
        // Adpate la box In
        bool  SetBoxOut(const Box2di & aBox) ;
        // Fait les init necessitant des appels virtuels, eq :
        // Calcule la grille (qui ne tient pas compte des decalage)
        void   VirtualPostInit() ;



       // Space est l'espace "commun de fusion"

       Pt2dr  ToFusionSpace(const Pt2dr &) const;  
       Pt2dr  FromFusionSpace(const Pt2dr &) const;

       void DoChIn();
       int  MaxChOut() ;
       const  std::vector<cChannelIn *> & ChIn();

       const std::string &        NameCorrige() const;
       const std::string &        NameGeomCorrige() const;
       virtual ~cCC_OneChannel();

       Box2dr  BoxIm2BoxFus(const Box2dr &) ;
       Box2dr  GlobBoxFus() ;

   protected :

       Pt2di SzCurInput() const;
       const cImageCmpCol &       mICC;
       cCC_Appli &                mAppli;
       std::string          mDir;;
       std::string          mNameInit;
       std::string          mNameCorrige;
       std::string          mNameGeomCorrige;

       std::string          mNameFile;
       Tiff_Im              mFileIm;
       GenIm::type_el       mTypeIn;
       Pt2di                mP0GlobIn;
       Pt2di                mP1GlobIn;
       Pt2di                mCurP0In;
       Pt2di                mCurP1In;


       CamStenope *         mCam;
       cDbleGrid  *         mGrid;
       Pt2dr                mOriFus;
       double               mScaleF;
       bool                 mChInMade;
       std::vector<cChannelIn *> mChIns;
       int                       mMaxChOut;
	double mFactF2C2;

   private :
       Pt2dr Direct(Pt2dr aP) const;
       bool OwnInverse(Pt2dr & aP) const;

};

class cCC_OneChannelSec : public cCC_OneChannel
{
   public :
       cCC_OneChannelSec
       (
            double aScale,
            const cImSec &,
	    cCC_OneChannel &   aMaster,
	    cCC_Appli &
       );
       void TestVerif();
       virtual ~cCC_OneChannelSec();

   private :
       Pt2dr Direct(Pt2dr aP) const;
       bool OwnInverse(Pt2dr & aP) const;

       const cImSec &  mIS;
       std::string     mNameCorresp;
       cCC_OneChannel&  mMaster;

       cElHomographie  mH_M2This;
       cElHomographie  mH_This2M;
       cDbleGrid  *         mGM2This;   //  Eventuellement la transfo sous forme grille
                                        // qui envoie vers la maitresse
};



class cChanelOut
{
   public :
      double  CalibRel(const cChanelOut  & , double aMaxRatio,double&  aSigma) const;
      cChanelOut(Pt2di aSz);
      void ResizeAndReset(const Pt2di& aSz);

      Im2DGen * Im();
      double GetVal(const Pt2di &) const;

      void AddChIn(cChannelIn *);

      double  GetInterpolPFus(const Pt2dr & aPFus,bool & Ok);
      // Pour l'instant ca s'appelle "modestement" initialisation par
      // interpolation car on envisage "un jour" d'implanter une approche
      // par assimilation  (pb inverse)
      void InitByInterp();
      const  std::vector<cChannelIn *>  & ChIns();
      virtual ~cChanelOut();

   private :
       double                    mSPds;
       int                       mNbCh;
       Pt2di                     mSz;
       std::vector<cChannelIn *> mChIns;
       Im2DGen *                 mIm;
       Im2D_Bits<1>              mMasq;
       TIm2DBits<1>              mTM;
};




class cCC_Appli
{
     public :
         cCC_Appli(const cCreateCompColoree &,int argc,char ** argv);


	 const std::string & WorkDir();
	 const cCreateCompColoree & CCC();

	 cInterfChantierNameManipulateur * ICNM();
	 const Box2dr &   BoxGlob() const;
         cCC_OneChannel&  Master();
         virtual ~cCC_Appli();

         void DoCalc();
         void DoStat();
         void DoMapping(int argc,char ** argv);
         bool  ModeMaping() const;
     private :
         void DoOneBoxOut(const Box2di & aBoxOut,const Box2di & aBoxSauvOut);


         void CreateOrUpdateInOut(const Box2di & aBoxOut);
         void InitByInterp();
	 void SauvImg(const Box2di & aBoxOut,const Box2di & aBoxSauvOut);
	 void SauvImg(const Box2di & aBoxOut,const Box2di & aBoxSauvOut,const cResultCompCol & aRCC);




         cChanelOut &   KCanOut(int aK);
         Fonc_Num       KF(int aK);


         std::string                         mWorkDir;
         std::string                         mComInit;
         cCreateCompColoree  mCCC;
	 cInterfChantierNameManipulateur * mICNM;
	 double              mScF;
	 int                 mMaxChOut;
	 int                 mNbChOut;

         bool                                mModeMaping;
         bool                                mInOutInit;

	 cCC_OneChannel                      *mMaster;

         std::vector<cCC_OneChannelSec *>    mVSecInit;
         std::vector<cCC_OneChannel *>    mVAllsInit;


         std::vector<cCC_OneChannelSec *>    mCurSecs;
         std::vector<cCC_OneChannel    *>    mCurAllCh;
	 int                                 mNbChAll;

	 std::vector<cChanelOut *>           mChOut;
	 std::string                         mNameMasterTiffIn;

         Box2di                              mBoxEspaceFus;
         Box2di                              mBoxEspaceCalc;
         int                                 mKBox;
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
