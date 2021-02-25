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
#include "StdAfx.h"

#define DEF_OFSET -12349876

class cCEM_OneIm;
class cCEM_OneIm_Epip;
class cCoherEpi_main;
class cCEM_OneIm_Nuage;


template <class Type> Type DebugMess(const std::string & aMes,const Type & aVal)
{
   std::cout << aMes << "\n";
   return aVal;
}


class cBoxCoher
{
     public :
         cBoxCoher(const Box2di & aBoxIn, const Box2di & aBoxOut, const std::string & aPost) :
              mBoxIn  (aBoxIn),
              mBoxOut (aBoxOut),
              mPost   (aPost)
         {
         }

         Box2di mBoxIn;
         Box2di mBoxOut;
         std::string mPost;
};


class cCEM_OneIm
{
     public :

          // Initialisation faisant appel a du virtuel
          void PostInit();
          virtual Pt3dr To3d(const Pt2di &,const Pt3dr * aVerif) const = 0;

          bool Empty() const;
          cCEM_OneIm (cCoherEpi_main * ,const std::string &,const Box2di & aBox,bool Visu,bool IsFirstIm);
          Box2dr BoxIm2(const Pt2di & aSzIm2);
          Box2dr BoxIm2(const Pt2di & aSzIm2,bool & OneOk);
          void SetConj(cCEM_OneIm *);
          virtual void UsePack(const ElPackHomologue &) ;

          Pt2dr ToIm2Gen(const Pt2dr & aP) const
          {
                bool Ok;
                return RoughToIm2(aP,Ok)- mConj->mRP0;
          }

          inline Pt2dr ToImCam(const Pt2dr & aP) const;

          Pt2dr ToIm2(const Pt2dr & aP,bool &Ok)
          {
                 Ok = IsOK(round_ni(aP));
                 if (Ok)
                    return RoughToIm2(aP,Ok)- mConj->mRP0;
                 else
                    return Pt2dr(0,0);
          }


          Pt2dr AllerRetour(const Pt2dr & aP,bool & Ok)
          {
                Pt2dr Aller = ToIm2(aP,Ok);
                if (!Ok) return aP;
                return mConj->ToIm2(Aller,Ok);
          }
          Im2D_U_INT1  ImAR();
          const Pt2di &  Sz() const {return mSz;}

          void VerifIm(Im2D_Bits<1> aMasq);
          Im2D_REAL4 VerifProf(Im2D_Bits<1> aMasq);
          void ComputeOrtho();

          virtual Im2D_REAL4 ImPx()
          {
                 ELISE_ASSERT(false,"ImPx");
                 return Im2D_REAL4(1,1);
          }
          virtual Im2D_INT2 ImPx_u2()
          {
                 ELISE_ASSERT(false,"ImPx");
                 return Im2D_INT2(1,1);
          }

          Im2D_REAL4   Im() {return     mIm;}

          double ResolAlti() const {return mResolAlti;}

          Video_Win *  Win()
          {
               // ELISE_ASSERT(mWin!=0,"cCEM_OneIm::Win()");
               return    mWin;
          }

     protected :
          virtual  Pt2dr  RoughToIm2(const Pt2dr & aP,bool & Ok) const = 0;
          bool  IsOK(const Pt2di & aP)
          {
              return mTMasq.get(aP,0) != 0;
          }

          Output VGray() {return mWin ?  mWin->ogray() : Output::onul(1) ;}
          Output VDisc() {return mWin ?  mWin->odisc() : Output::onul(1) ;}


          bool             mIsFirstIm;
          cCoherEpi_main * mCoher;
          cCpleEpip *      mCple;
          std::string      mDir;
          std::string      mDirM;
          std::string      mNameNuage;
          cXML_ParamNuage3DMaille mParNuage;
          double                  mResolAlti;
          std::string      mNameInit;
          std::string      mNameImMatched;
          std::string      mNameFinal;
          Tiff_Im          mTifIm;
          Box2di           mBox;
          Pt2di            mSz;
          Pt2di            mP0;
          Pt2dr            mRP0;
          Im2D_REAL4       mIm;
          double           mVMaxIm;
          Im2D_REAL4       mImOrtho;
          double           mVMaxOrtho;


          Video_Win *      mWin;
          Video_Win *      mWin2;
          cCEM_OneIm *     mConj;
          Im2D_Bits<1>     mImMasq;
          TIm2DBits<1>     mTMasq;

          // DEBUGM3D
          std::string       mNNOri;
          cElNuage3DMaille*  mNuOri;
};

class cCEM_OneIm_Epip  : public cCEM_OneIm
{
    public :
          virtual Pt3dr To3d(const Pt2di &,const Pt3dr * aVerif) const ;
          Im2D_REAL4 ImPx() {return mImPx;}
          Im2D_INT2  ImPx_u2() {return mImPx_u2;}

          cCEM_OneIm_Epip (cCoherEpi_main * ,const std::string &,const Box2di & aBox,bool Visu,bool IsFirstIm,bool Final);

          virtual  Pt2dr  RoughToIm2(const Pt2dr & aP,bool & Ok) const
          {
             Ok = true;
             return Pt2dr(aP.x+mTPx.getprojR(aP),aP.y) + mRP0;
          }
          void UsePack(const ElPackHomologue &) ;

          std::string      mNamePx;
          Tiff_Im          mTifPx;
          Im2D_REAL4       mImPx;
          TIm2D<REAL4,REAL8> mTPx;
          Im2D_INT2         mImPx_u2;

          std::string      mNameMasq;
          Tiff_Im          mTifMasq;
          CamStenope *     mCam;

};

class cCEM_OneIm_Nuage  : public cCEM_OneIm
{
      public :
          cCEM_OneIm_Nuage (cCoherEpi_main * ,const std::string &,const std::string &,const Box2di & aBox,bool Visu,bool IsFirstIm);
          virtual Pt3dr To3d(const Pt2di &,const Pt3dr * aVerif) const ;
      private :
          Pt2dr  RoughToIm2(const Pt2dr & aP,bool & Ok) const
          {
                 ELISE_ASSERT(false,"See RP0 ???? ");
                 if (! mNuage1->IndexIsOKForInterpol(aP))
                 {
                      Ok = false;
                      return Pt2dr (0,0);
                 }
                 Pt3dr aP3 = mNuage1->PtOfIndexInterpol(aP);
                 Pt2dr aRes = mNuage2->Terrain2Index(aP3);

                 Ok = true;
                 return aRes;
          }

          std::string              mDirLoc1;
          std::string              mDirLoc2;
          std::string              mDirNuage1;
          std::string              mDirNuage2;
          cParamModifGeomMTDNuage  mPGMN;
          std::string              mNameN;
          cXML_ParamNuage3DMaille  mParam1;
          cElNuage3DMaille *       mNuage1;
          cXML_ParamNuage3DMaille  mParam2;
          cElNuage3DMaille *       mNuage2;
};




class cOneContour
{
     public :
        ElList<Pt2di> *  mL;
        double           mSurf;
        bool             mExt;
};
inline bool operator < (const cOneContour & aC1,const cOneContour & aC2)
{
   return aC1.mSurf > aC2.mSurf;
}

class cCoherEpi_main : public Cont_Vect_Action
{
     public :


        void action(const  ElFifo<Pt2di> & aFil,bool                ext);


        friend class cCEM_OneIm;
        friend class cCEM_OneIm_Epip;
        friend class cCEM_OneIm_Nuage;
        cCoherEpi_main (int argc,char ** argv);

        std::string NameIm(bool First);
        cInterfChantierNameManipulateur * ICNM();
        cMasqBin3D * Masq3d();
        const std::string & Ori() const;
        



     private  :
        Box2di       mBoxIm1;
        int          mSzDecoup;
        int          mBrd;
        Pt2di        mIntY1;
        std::string  mNameIm1;
        std::string  mNameIm2;
        std::string  mOri;
        std::string  mDir;
        cInterfChantierNameManipulateur * mICNM;
        cMasqBin3D   * mMasq3d;
        cCpleEpip *  mCple;
        bool          mWithEpi;
        bool          mNoOri;
        bool          mByP;
        bool          mInParal;
        bool          mCalledByP;
        std::string   mPrefix;
        std::string   mPostfixP;
        cCEM_OneIm  * mIm1;
        cCEM_OneIm  * mPIm2;  // Renomed mPIm2 as it can be 0 when empty zone

        int           mDeZoom;
        int           mNumPx;
        int           mNumMasq;
        bool          mVisu;
        double        mSigmaP;
        bool          mRegulCheck;
        bool          mFilterBH;
        double        mStep;
        double        mRegul;
        double        mReduceM;
        bool          mFinal;
        Im2D_REAL4    mImQualDepth;
        bool          mDoMasq;
        double        mDoMasqSym;
        bool          mUseAutoMasq;
        std::vector<cOneContour> mConts;
        double        mBSHRejet;  // Def 0.02
        double        mBSHOk;     // Def 2 * mBSHMin
};

class cQual_DeptMap
{
   public :
       cQual_DeptMap(Im2D_REAL4,Im2D_Bits<1>);
   private :
       TIm2D<REAL4,REAL8>  mTProf; 
       TIm2DBits<1>        mTMasq;
};

Im2D_REAL4 ImageQualityGrad(Im2D_REAL4 aProf,Im2D_Bits<1> aMasq,Video_Win *);

Pt2dr cCEM_OneIm::ToImCam(const Pt2dr & aP) const
{
    return (aP+mRP0) * mCoher->mDeZoom;
}


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
