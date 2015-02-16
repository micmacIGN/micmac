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

   Cas d'utilisation des espaces :

      - Ful Im1  - Ful Im2  

      - Ful Im1 Decoupe  -  Im2 asservi

      - Box Im1  - Box Im2

      - Box Im1  - Im2 asservi

     Cas incompatible :
         - Un decoupage,
         - Plusieurs images
         - pas d'asservissement

     Cas incompatible :
         - asservissement avec BoxIm (caso


        - Decoupage + 
*/

#ifndef _ELISE_DIGEO_H_
#define _ELISE_DIGEO_H_


#include "general/all.h"
#include "private/all.h"
#include "im_tpl/image.h"

#include "cParamDigeo.h"

#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;


//  cRotationFormelle::AddRappOnCentre

namespace NS_ParamDigeo
{

//   Represente un image d'une octave a une resolution donnee
class cImInMem;
template <class Type> class cTplImInMem ;

//  Represente une octave !! Contient autant cImInMem que de sigma de gaussienne
class cOctaveDigeo;
template <class Type> class cTplOctDig ;

//   Represente une image ; contient autant d'octave que necessaire
class cImDigeo;


/*
    Class virtuelle , d'interface avec les classe permettant de correler rapidemnt
  selon une ligne.

        virtual void Convol(Type * Out,Type * In,int aK0,int aK1) :

             - (A faire) : methode par defaut, correlation standar

        static cConvolSpec<Type> * Get(tBase* aFilter,int aDeb,int aFin,int aNbShitX,bool ForGC);
        cConvolSpec(tBase* aFilter,int aDeb,int aFin,int aNbShitX,bool ForGC);
*/
template <class Type> class cConvolSpec;

class cAppliDigeo;

class cVisuCaracDigeo;

const int PackTranspo = 4;

typedef enum
{
  eTES_instable,
  eTES_GradFaible,
  eTES_TropAllonge,
  eTES_Ok
} eTypeExtreSift;


/*****************************************************************/
/*                                                               */
/*   Fonctions elementaires de noyaux 1D faits notamment pour    */
/* permettre le calcul de noyau enchaine                         */
/*                                                               */
/*****************************************************************/

//  Resout l'equation  aI o I2 = aI3 , au sens des moindres carres
Im1D_REAL8 DeConvol
(
     int aC2,   // Indexe 0 dans I2
     int aSz2,  // Taille I2
     Im1D_REAL8 aI1,   // Kernel 1
     int aC1,          // Indexe 0 dans I1
     Im1D_REAL8 aI3,   // Kernel 3
     int aC3           // Indexe 0 dans I3
);
// Paramametrage standardA  0 en centre image
Im1D_REAL8 DeConvol(int aDemISz2,Im1D_REAL8 aI1,Im1D_REAL8 aI3);



// Convolution C1 et C2 = indexe 0,  lent ; pour verif DeConvol
Im1D_REAL8 Convol(Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI2,int aC2);
// Parametrage stantdard 
Im1D_REAL8 Convol(Im1D_REAL8 aI1,Im1D_REAL8 aI2);

// Force l'image à une integrale donnee
Im1D_REAL8 MakeSom(Im1D_REAL8 aIm,double aSomCible);
Im1D_REAL8 MakeSom1(Im1D_REAL8 aIm);

//  Calcul un noyau gaussien en faisant pour chaque pixel la valeur integrale.
Im1D_REAL8  GaussianKernel(double aSigma,int aNb,int aSurEch);

//  Calcul le nombre d'element pour que la gaussiennne puisse etre tronquee a Residu pres
int NbElemForGausKern(double aSigma,double aResidu);

//  Calcule un noyau gaussien
Im1D_REAL8  GaussianKernelFromResidu(double aSigma,double aResidu,int aSurEch);


// Conversion d'un noyau double (de somme 1) en entier, en conservant la somme
// (Pour une image entiere qui sera interpretee comme rationnele de quotient aMul)
Im1D_INT4 ToIntegerKernel(Im1D_REAL8 aRK,int aMul,bool aForceSym);





     //     Pour representer une image

class cImInMem
{
     public :

         GenIm::type_el  TypeEl() const;
         int  RGlob() const;
         double ROct() const;
         Pt2di Sz() const;
         cImInMem *  Mere();
         cOctaveDigeo & Oct();
         void  SetMere(cImInMem *);
         void SauvIm(const std::string & = "");

         // void MakeReduce_121();

         virtual void VMakeReduce_121(cImInMem &)=0;
         virtual void VMakeReduce_010(cImInMem &)=0;
         virtual void VMakeReduce_11(cImInMem &)=0;

         void MakeReduce(cImInMem &,eReducDemiImage);


         // virtual void Resize(const Pt2di & aSz) = 0;
         virtual void LoadFile(Tiff_Im aFile,const Box2di & aBox) = 0;
         virtual Im2DGen Im() = 0;

         // La relation mere-fille a meme DZ se fait entre image de mm type
         // virtual void  SetMereSameDZ(cImInMem *)=0;

         // virtual void MakeConvolInit(double aSigm )= 0;
         virtual void ReduceGaussienne() = 0;

         virtual double CalcGrad2Moy() = 0;

     protected :

         cImInMem(cImDigeo &,const Pt2di & aSz,GenIm::type_el,cOctaveDigeo &,double aResolOctaveBase,int aKInOct,int IndexSigma);
         cAppliDigeo &    mAppli;
         cImDigeo &       mImGlob;
         cOctaveDigeo &   mOct;
         Pt2di            mSz;
         GenIm::type_el   mType;
         int              mResolGlob;
         double           mResolOctaveBase;
         int              mKInOct;
         int              mIndexSigma;
         int              mNbShift;
         cImInMem *       mMere;
         cImInMem *       mFille;
     

         Im1D_REAL8 mKernelTot;  // Noyaux le reliant a l'image de base de l'octave
     private :
        cImInMem(const cImInMem &);  // N.I.
};


template <class Type> class cTplImInMem : public cImInMem
{
     public :
        typedef typename El_CTypeTraits<Type>::tBase tBase;
        typedef Im2D<Type,tBase>   tIm;
        typedef TIm2D<Type,tBase>  tTIm;
        typedef cTplImInMem<Type>  tTImMem;

        cTplImInMem(cImDigeo &,const Pt2di & aSz,GenIm::type_el,cTplOctDig<Type> &,double aResolOctaveBase,int aKInOct,int IndexSigma);


        void SetConvolSepXY
             (
                  const cTplImInMem<Type> & aImIn,
                  Im1D<tBase,tBase> aKerXY,
                  int  aNbShitXY
             );


        //tTIm  & TIm() {return TIm;}
        //const tTIm  & TIm() const {return TIm;}
        tIm  TIm() const {return mIm;}
        void LoadFile(Tiff_Im aFile,const Box2di & aBox) ;
        bool InitRandom();

        void VMakeReduce_121(cImInMem &);
        void VMakeReduce_010(cImInMem &);
        void VMakeReduce_11(cImInMem &);
        void Resize(const Pt2di & aSz);
        double CalcGrad2Moy();
        Im2DGen Im() ;
        void  SetMereSameDZ(cTplImInMem<Type> *);
        void  SetOrigOct(cTplImInMem<Type> *);
        // void MakeConvolInit(double aSigm );
        void ReduceGaussienne();

        void ExtractExtremaDOG
             (
                   const cSiftCarac & aSC,
                   cTplImInMem<Type> & aPrec,
                   cTplImInMem<Type> & aNext1,
                   cTplImInMem<Type> & aNext2
             );
     private :

        eTypeExtreSift CalculateDiff(Type***aC,int anX,int anY,int aNiv);

/*
        SetConvolBordX :
 
          Pour la "colonne" X, calcul dans ImOut toute les convolution en gerant 
       les effets de bord :

       aDebX  , aFinX : borne du filtre, typiquement de -SzKer , + SzKer (inclus)

  Utilise 
  
template <class tBase> tBase ClipForConvol(int aSz,int aKXY,tBase * aData,int & aDeb,int & aFin) :

    Clip l'intervalle (genre [-SzKe,+SzK] au depart) pour que la convol ne deborde 
    pas de [0,aSz[


inline tBase CorrelLine(tBase aSom,const Type * aData1,const tBase *  aData2,const int & aDeb,const int & aFin)

    Produit scalaire tout a fait basique, utilise pour correler les bord


*/


        static void SetConvolBordX
             (
                  Im2D<Type,tBase> aImOut,
                  Im2D<Type,tBase> aImIn,
                  int aX,
                  tBase *,int DebX,int aFinX
             );

        static void SetConvolSepX
             (
                  Im2D<Type,tBase> aImOut,
                  Im2D<Type,tBase> aImIn,
                  tBase *,int DebX,int aFinX,
                  int  aNbShitX,
                  cConvolSpec<Type> *
             );




        void SetConvolSepX
             (
                  const cTplImInMem<Type> & aImIn,
                  tBase *,int DebX,int aFinX,
                  int  aNbShitX,
                  cConvolSpec<Type> *
             );


        void SelfSetConvolSepY
             (
                  tBase *,int DebY,int aFinY,
                  int  aNbShitY,
                  cConvolSpec<Type> *
             );

         template <class TMere> void  MakeReduce_121(const cTplImInMem<TMere> &);
         template <class TMere> void  MakeReduce_010(const cTplImInMem<TMere> &);
         template <class TMere> void  MakeReduce_11(const cTplImInMem<TMere> &);

         std::string NameClassConvSpec(tBase* aFilter, int aDeb, int aFin);
         void MakeClassConvolSpec(FILE *,FILE *,tBase* aFilter,int aDeb,int aFin,int aNbShit); 


         void  ExtramDOG(Type *** aC,const Pt2di & aP,bool & isMax,bool & isMin);
         bool  SupDOG(Type *** aC,const Pt3di& aP1,const Pt3di& aP2);
         tBase DOG(Type *** aC,const Pt3di& aP1);



         cTplOctDig<Type> & mTOct;
         tIm    mIm;
         tTIm   mTIm;
         tTImMem *  mTMere;
         tTImMem *  mTFille;
         tTImMem *  mOrigOct;
         Type **    mData;
         tBase      mDogPC;  // Dif of Gauss du pixel courrant

         
     private :
          cTplImInMem(const cTplImInMem<Type> &);  // N.I.
          void ExploiteExtrem(int anX,int anY);


          static tBase  ** theMDog;

          double            mSeuilTr2Det;
          double            mSeuilGrad;
          int               mBrd;
          Pt2dr             mP;
          double            mGX;
          double            mGY;
          double            mDxx;
          double            mDyy;
          double            mDxy;
          double            mTrX;
          double            mTrY;

          eTypeExtreSift    mResDifSift;
          int               mNbExtre;
          int               mNbExtreOK;
};


class cOctaveDigeo
{
    public :
        static cOctaveDigeo * Alloc(GenIm::type_el,cImDigeo &,int aNiv,Pt2di aSzMax);

        int NbIm() const;
        cImInMem * KthIm(int aK) const;
        int                      Niv() const;

        // void AddIm(cImInMem *);

        virtual cImInMem * AllocIm(double aResolOctaveBase,int aK,int IndexSigma) = 0;
        virtual cImInMem * GetImOfSigma(double aSig) = 0;
        virtual  cImInMem * ImBase() = 0;
        void SetNbImOri(int aNbIm);
        int  NbImOri() const;

        virtual void DoSiftExtract(const cSiftCarac &) = 0;
        virtual void PostPyram() = 0;
    protected :
        cOctaveDigeo(GenIm::type_el,cImDigeo &,int aNiv,Pt2di aSzMax);


        GenIm::type_el           mType;
        cImDigeo &               mIm;
        int                      mNiv;
        std::vector<cImInMem *>  mVIms;
        Pt2di                    mSzMax;
        int                      mNbImOri;  // de NbByOctave()
     private :
        cOctaveDigeo(const cOctaveDigeo &);  // N.I.
};


template <class Type> class cTplOctDig  : public cOctaveDigeo
{
    public :
         cTplOctDig(GenIm::type_el,cImDigeo &,int aNiv,Pt2di aSzMax);
         cImInMem * AllocIm(double aResolOctaveBase,int aK,int IndexSigma);
         cImInMem * GetImOfSigma(double aSig);

         Type*** Cube();

         cImInMem * ImBase();
         cTplImInMem<Type> * TypedImBase();

    private :

         void DoSiftExtract(const cSiftCarac &) ;
         void PostPyram() ;
         cTplImInMem<Type> * AllocTypedIm(double aResolOctaveBase,int aK,int IndexSigma);
         cTplImInMem<Type> * TypedGetImOfSigma(double aSig);

        std::vector<cTplImInMem<Type> *>  mVTplIms;


        std::vector<Type **>  mVDatas;
        Type ***              mCube;
        cTplImInMem<Type> *  mImBase;
    private :
        cTplOctDig(const cTplOctDig<Type> &);  // N.I.
};



class cImDigeo
{
     public :
         cImDigeo
         (
              int aNum,
              const cImageDigeo &,
              const std::string & aName,
              cAppliDigeo &
         );
        // void ComputeCarac();
        const std::string  &  Name() const;
        cAppliDigeo &  Appli();
        Box2di BoxIm() const;

 // Pour pouvoir se dimentionner au "pire" des cas, chaque image est
 // d'abord notifiee de l'existence d'une box
        void NotifUseBox(const Box2di &);
        void AllocImages();
        void LoadImageAndPyram(const Box2di & aBox);
        void DoCalcGradMoy(int aDZ);



       void DoExtract();
       const cImageDigeo &  IMD();
       cVisuCaracDigeo  *  CurVisu();
       cOctaveDigeo & GetOctOfDZ(int aDZ); 
       cOctaveDigeo * SVPGetOctOfDZ(int aDZ); 


       void SetDyn(double);
       double Dyn() const;
       double G2Moy() const;
     private :


        void DoSiftExtract();

        GenIm::type_el  TypeOfDeZoom(int aDZ) const;

        std::string                   mName;
        cAppliDigeo &                 mAppli;
        const cImageDigeo &           mIMD;
        int                           mNum;
        std::vector<cImInMem *>       mVIms;
        Tiff_Im *                     mTifF;
        Pt2di                         mSzGlob;
        Box2di                        mBoxIm;
        Pt2di                         mSzCur;
        std::vector<cOctaveDigeo *>   mOctaves;
        Pt2di                         mSzMax;
        int                           mNiv;
        cVisuCaracDigeo  *  mVisu;

        bool                         mG2MoyIsCalc;
        double                       mG2Moy;
        double                       mDyn;
     private :
        cImDigeo(const cImDigeo &);  // N.I.
        
};

template <class Type> class cConvolSpec
{
    public :
        typedef typename El_CTypeTraits<Type>::tBase tBase;

        virtual void Convol(Type * Out,Type * In,int aK0,int aK1) ;
        static cConvolSpec<Type> * Get(tBase* aFilter,int aDeb,int aFin,int aNbShitX,bool ForGC);
        cConvolSpec(tBase* aFilter,int aDeb,int aFin,int aNbShitX,bool ForGC);
    protected :

    private :
        bool Match(tBase *  aDFilter,int aDebX,int aFinX,int  aNbShitX,bool ForGC);
        static std::vector<cConvolSpec<Type> *>   theVec;
        

        int mNbShift;
        int mDeb;
        int mFin;
        std::vector<tBase>  mCoeffs;
        bool mForGC;
        

        cConvolSpec(const cConvolSpec<Type> &); // N.I.
};


class cVisuCaracDigeo
{
     public :
        cVisuCaracDigeo(cAppliDigeo &,Pt2di aSz,int aZ,Fonc_Num aF,const cParamVisuCarac &); 
        void Save(const std::string&);
        void SetPtsCarac 
             (
                 const Pt2dr & aP,
                 bool aMax,
                 double aSigma,
                 int  aIndSigma,
                 eTypeExtreSift
                 
             );
     private :
        cAppliDigeo &      mAppli;
        Pt2di              mSz1;
        int                mZ;
        Pt2di              mSzZ;
        Im2D_U_INT1        mIm;
        TIm2D<U_INT1,INT>  mTIm;
        const cParamVisuCarac & mParam;
        int          mDynG;
};


class cAppliDigeo : public cParamDigeo
{
    public : 
       cAppliDigeo
       ( 
              cResultSubstAndStdGetFile<cParamDigeo> aParam,
              cAppliDigeo *                          aMaterAppli,
              cModifGCC *                            aModif,
              bool                                   IsLastGCC

       );

        void DoAll();
        const std::string & DC() const;
        cInterfChantierNameManipulateur * ICNM();


       FILE *  FileGGC_H();
       FILE *  FileGGC_Cpp();
       bool    MultiBloc() const;

       cModifGCC *      ModifGCC() const;



    private :
       void DoCarac();
       
       static void InitConvolSpec();


       void AllocImages();

       std::string                       mDC;
       cInterfChantierNameManipulateur * mICNM;
       std::vector<cImDigeo *>           mVIms;

       cAppliDigeo *                     mMaster;
       cModifGCC *                       mModifGCC;
       bool                              mLastGCC;

       FILE *  mFileGGC_H;
       FILE *  mFileGGC_Cpp;

     private :
        cAppliDigeo(const cAppliDigeo &);  // N.I.
        

};

#define InstantiateClassTplDigeo(aClass)\
template  class aClass<U_INT1>;\
template  class aClass<U_INT2>;\
template  class aClass<REAL4>;\
template  class aClass<INT>;


};

#endif //  _ELISE_DIGEO_H_




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
