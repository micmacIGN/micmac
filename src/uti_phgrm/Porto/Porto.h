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

class cOneImOrhto;
class cLoadedIm;
class cAppli_Ortho;

#define LabelNoIm  10000
#define MaxPC      255

 //============================================

class cOneImOrhto
{
    public :
       friend class cLoadedIm;

       cOneImOrhto
       (
             int anInd,
             cAppli_Ortho &,
             const std::string & aName,
             const cMetaDataPartiesCachees &,
             const Box2di & aBox
       );
       Tiff_Im   TF();

       bool Instersecte(const Box2di &  aBoxIn);
       const std::string & Name() const;
       const Pt2di &    SzIm() const;
       const int & TheInd() const;

       void SetERIPrinc(cER_OneIm *);
       cER_OneIm *    ERIPrinc();
       void SetERIPhys(cER_OneIm *);
       cER_OneIm *    ERIPhys();

       void DoSimul(const cSectionSimulImage & aSSI);

       double Priorite() const;
       bool   Im2Test() const;

    private :
       cOneImOrhto(const cOneImOrhto &); // N.I.
       const std::string & WD() const;
       const cCreateOrtho & CO() const;
       cInterfChantierNameManipulateur * ICNM() const;

       int            mInd;
       cAppli_Ortho & mAppli;
       std::string    mName;
       Tiff_Im        mTif;
       Pt2di          mSzIm;
       cMetaDataPartiesCachees mMdPC;
       Tiff_Im        mTifPC;
       Tiff_Im        mTifIncH;
       Box2di         mBox;
       cER_OneIm *    mERIPrinc;
       cER_OneIm *    mERIPhys;
       double         mPriorite;
       bool           mIm2Test;
};



class cLoadedIm
{
    public :
       friend class cOneImOrhto;
       cLoadedIm (cAppli_Ortho & ,int aInd,Tiff_Im & aTF0,const Pt2di aSz);
       bool Init(cOneImOrhto &,const Box2di & aBoxOut,const Box2di & aBoxIn);
       void TransfertIm(const std::vector<Im2DGen *> &,const Pt2di& aPLoc,const Pt2di& PGlob);

       void UpdateNadirIndex(Im2D_REAL4  aScoreNadir,Im2D_INT2 aIndexNadir);
  
       int ValeurPC(const Pt2di &) const;
       Im2D_U_INT1  ImPCBrute();
       Fonc_Num     FoncInc();  // Fonction d'incidence
       int Ind() const;
       cOneImOrhto *   CurOrtho();
       // Dec entre le loaded et l'origine de la sous ortho
       const Pt2di&  DecLoc() const;
       std::vector<double> Vals(const Pt2di &) const;
       double ValCorrel(const Pt2di & aP) const;
       bool   OkCorrel(const Pt2di & aP,int  aSzV) const;

       double Correl(const Pt2di & aP,cLoadedIm *,const  std::vector<std::vector<Pt2di> > &,int aSz);

       void TestDiff(cLoadedIm *);
       bool   Im2Test() const;
    private :
       cLoadedIm(const cLoadedIm &) ; // N.I. 

       cAppli_Ortho &  mAppli;
       cOneImOrhto *          mCurOrtho;
       int                     mInd;
       Im2D_REAL4              mImIncH;  // +ou- image du nadir
       Im2D_U_INT1             mImPCBrute;  //  Partie cachee (ou simple Masque ?)
       Im2D_U_INT1             mImPC;  //  Partie cachee (ou simple Masque ?)
       TIm2D<U_INT1,INT>       mTImPC;
       std::vector<Im2DGen *>  mIms;
       int                     mChCor;
       Pt2di                   mSz;
       Pt2di                   mSzRed;
       Pt2di                   mDecLoc;

       double                  mFactRed;
       double                  mIncMoy;
       Pt2di                   mRedDecLoc;
       Pt2dr                   mPR_RDL; // Partie Reel Reduite Decalage loc
       
};




class cAppli_Ortho
{
     public :
         typedef enum
         {
             eModeOrtho,
             eModeCompMesSeg
         } eModeMapBox;

         cAppli_Ortho(const cCreateOrtho &,int argc,char ** argv);

         void DoAll();

         cInterfChantierNameManipulateur * ICNM() const;
         const cCreateOrtho & CO() const;
         const std::string & WD() const;

         Video_Win  *   W();
         bool DoEgGlob() const;

         double Correl(cLoadedIm * aIm1,cLoadedIm * aIm2);


         bool ValMasqMesure(const Pt2di & aP)
         {
              return ( mTImMasqMesure.get(aP,1)!=0 );
         }

         void SetRMax(double aRad,int aK)
         {
             ElSetMax(mRadMax[aK],aRad);
         }

         double      DynGlob() const 
         {
              return   mDynGlob;
         }
     private :

         Box2di  BoxImageGlob();
         cFileOriMnt GetOriMnt(const std::string & aName) const;

         cFileOriMnt * GetMtdMNT();
         std::vector<cMetaDataPartiesCachees>  * VMDPC();

         void AddOneMasqMesure(Fonc_Num & aFMGlob,int & aCpt,const cMasqMesures & aMM,const Box2di & aBoxIn);
         void ResetMasqMesure(const Box2di & aBoxIn);

         void DoOrtho();
         void MapBoxes(eModeMapBox);

         void DoOneBox(const Box2di& aBoxOut,const Box2di& aBoxIn,eModeMapBox);

         void OrthusCretinus();
         void OrthoRedr();
         void Resize(const Pt2di &);
           
         void DoEgalise();
         void ComputeMesureEgale();
         void RemplitOneStrEgal();

         void DoIndexNadir();
         void MakeOrthoOfIndex();
         void InitInvisibilite(const cBoucheTrou &);
         void InitBoucheTrou(const Pt2di &,const cBoucheTrou &);
         void InitBoucheTrou(const cBoucheTrou &);
         Liste_Pts_INT2 CompConx(Pt2di aGerm,int aValSet);
         double ScoreOneHypBoucheTrou
                (
                     const cBoucheTrou  & aBT,
                     Liste_Pts_INT2 aL,
                     cLoadedIm * aLI
                );



         void SauvAll();
         void SauvOrtho();
         void SauvLabel();

         Im1D_U_INT2   InitRanLutLabel();
         void VisuLabel();

         const std::vector<std::string> * GetImNotTiled(const std::vector<std::string> *);




         std::string mWorkDir;
         cCreateOrtho mCO;
         cInterfChantierNameManipulateur * mICNM;
         const std::vector<std::string>  * mVIm;
         std::vector<cMetaDataPartiesCachees>  * mVMdpc;
         cFileOriMnt *                     mMtDMNT;
         Pt2di                             mSzMaxIn;

         std::vector<cOneImOrhto *> mVAllOrhtos;
         Tiff_Im *                  mTF0;
         Tiff_Im *                  mFileOrtho;
         Box2di                     mBoxCalc;
         Pt2di                      mSzCur;
         Box2di                     mCurBoxIn;
         Box2di                     mCurBoxOut;

         std::vector<cLoadedIm *> mVLI;
         std::vector<cLoadedIm *> mReserveLoadedIms;

         static const int         theNoIndex = -1;              

         Im2D_U_INT1              mImEtiqTmp;
         TIm2D<U_INT1,INT>        mTImEtiqTmp;

         Im2D_INT2                mImIndex;
         TIm2D<INT2,INT>          mTImIndex;
         Im2D_REAL4               mScoreNadir;
         TIm2D<REAL4,REAL8>       mTScoreNadir;

         Im1D_U_INT2              mLutLabelR;
         Im1D_U_INT2              mLutLabelV;
         Im1D_U_INT2              mLutLabelB;

         Im1D_U_INT2              mLutInd;

         Im2D_U_INT1              mImMasqMesure;
         TIm2D<U_INT1,INT>        mTImMasqMesure;

         std::vector<Im2DGen *>  mIms;
         Video_Win  *            mW;

         bool                         mEgalise;
         const cSectionEgalisation *  mSE;
         bool                         mCompMesEg;
         std::string                  mNameFileMesEg;
         cER_Global *                 mERPrinc;
         cER_Global *                 mERPhys;
         bool                         mDoEgGlob;
 
         std::vector<std::vector<Pt2di> >   mVoisCorrel;
         int                                mSzVC;
         double                             mSeuilCorrel;

         double                             mNbPtMoyPerIm;
         int                                mNbCh;
         std::vector<double>                mRadMax;
         double                             mDynGlob;
         int                                mNbIm2Test;
         int                                mNbLoadedIm2Test;
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
