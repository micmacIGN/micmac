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



#ifndef _ELISE_TRAITEMENT_RADIOM_  // general
#define _ELISE_TRAITEMENT_RADIOM_

// Classe d'interface, les classe "concrete" derivee sont definie dans
// un ".cpp".  Pour creer un objet il faut passer
// par les allocateur static


   class cEtalRelOneChan;
   class cSpecifEROC;
   class cValuesEROC;
   class cColorCalib;

class cER_MesureOneIm;
class cER_MesureNIm;
class cER_OneIm;
class cER_Global;
class cER_SysResol;
class cER_ParamOneSys;
class cER_SolOneCh;
class cER_SolOnePower;
class cElemGrapheIm;

/********************************************/
/*                                          */
/*         ETALONNAGE                       */
/*                                          */
/********************************************/


class EtalRelOneChan
{
    public :
};



/********************************************/
/*                                          */
/*         EGALISATION                      */
/*                                          */
/********************************************/


typedef enum
{
   eERG_2D_Pure,
   eERG_2D,
   eERG_3D,
   eERG_3DNorm
} eTypeERGMode;

class cER_MesureOneIm
{
     public :
         cER_MesureOneIm
         (
               int aKIm,
               const Pt2df& aPt,
               const std::vector<float> & aV
         );

         void  write(ELISE_fp &) const;
         static cER_MesureOneIm read(ELISE_fp &);
         int NBV() const;
         inline int KIm() const;
         inline const Pt2df & Pt() const;
         inline Pt2dr RPt() const;
         inline float KthVal(int aK) const;
         double SomVal() const;
         void AddMoyVal(std::vector<double> & aSomV,std::vector<double> & aSom1);
     private  :
         int    mKIm;
         Pt2df               mPtIm;
         std::vector<float>  mVal;
    
};

class cER_MesureNIm
{
    public :
        friend class cER_OneIm;
        //Abs et norm n'ont pas forecement de raison d'etre, mais selon le 
        // principe 1 seul consctructeur
        cER_MesureNIm(eTypeERGMode,const Pt3df& aPAbs,const Pt3df & aPNorm);

 
         void  write(ELISE_fp &) const;
         static void  read(cER_MesureNIm&,ELISE_fp &);

         int NBV() const;

         inline int NbMes() const;
         inline const cER_MesureOneIm & KthMes(int aK) const;
         void AddAMD(cAMD_Interf *);
         void AddMoyVal(std::vector<double> & aSomV,std::vector<double> & aSom1);
         Pt3dr  PAbs() const;
    private :
        void AddMesure(const cER_MesureOneIm &);
        U_INT1                       mMode;
        Pt3df                        mPAbs;
        Pt3df                        mPNorm;
        std::vector<cER_MesureOneIm> mMes;
};


class cER_SolOnePower
{
    public :
        cER_SolOnePower(const cER_SolOneCh & ,Pt2di aDeg,const std::vector<double> & aSol);
        double Value(const Pt2dr & aP) const;
    private :
        const cER_SolOneCh & mSolCh;
        const cER_OneIm    & mIm;
        int                  mNbDisc;
        RImGrid              mGrid;
        
};



class cER_SolOneCh
{
   public :
        const cER_OneIm    &  Im() const;
        cER_SolOneCh(int aKCh,const cER_OneIm & anIm,Im1D_REAL8 aSolGlob);
         ~cER_SolOneCh();
       
        inline double Value(const Pt2dr & aP,const std::vector<double> &) const;



   private :
        const cER_OneIm    &  mIm;
        std::vector<cER_SolOnePower *> mSols;
};

class cER_OneIm
{
     public :
         void PushCoeff(std::vector<double> &,double aSign);
         cER_OneIm
         (
             cER_Global * anERG,
             int aKIm,
             Pt2di aSz,
             const std::string & aName
         );
          // Les points sont normalises pour limiter la taille des
          // coefficients
          Pt2dr   ToPNorm(const Pt2df & aPIm) const;

         void AddMesure(cER_MesureNIm &,const Pt2di& aPt, const std::vector<float> &aV);

         const std::string & Name() const;
         const Pt2di  & SzIm() const;
         void  write(ELISE_fp &) const;
         static cER_OneIm * read(ELISE_fp & aFP,cER_Global *);

         // retourne l'adresse de l'element de deg 1 en rad et 0 en xy
         int InitObs(const Pt2df & aPI,double aV0,const cER_ParamOneSys & aParam);
         void MakeVecVals(std::vector<double>&Vals,const Pt2df&aPI,double aV0,const cER_ParamOneSys&aParam);

         cIncIntervale * II(const int &K) const;

         void PrevSingul(int aKR1XY0,double aPds,int aKCh,double aVal,cGenSysSurResol *);
         void AddRappelValInit(double aPds,int aKCh,double aVal,cGenSysSurResol *);
         void AddEqValsEgal(double aPds,int aKCh,cER_OneIm & ,cGenSysSurResol *);
         cER_Global * ERG() const;
  
         void SetSol(int aKCh,Im1D_REAL8 aSolGlob);

         double  Value(const Pt2dr& aP,double anInput,int aKParam);

         inline int KIm() const;

         bool UseForCompenseGlob() const;
         void ValueGlobCorr(const Pt2dr& aP, std::vector<double> & Out,const std::vector<double> & In,const Pt3dr& aPGlob);
         double  ValueGlobCorr (const Pt2dr& aP,double aVal ,int aKParam, const Pt3dr& aPGlob);

         void ValueGlobBrute(const Pt2dr& aP, std::vector<double> & Out,const std::vector<double> & In,const Pt3dr& aPGlob);
         double ValueGlobBrute(const Pt2dr& aP,double aVal,int aKParam,const Pt3dr& aPGlob);


         double FactCorrectL1(const Pt2dr &) const;
         void   SetParamL1(double aK0,double aKx,double aKy);

         void SetSeuilL1Predict(double aSeuil);
         double  SeuilPL1() const;
         int  &    NbMesValidL1() ;


     private :
         void ValueLoc(const Pt2dr& aP, std::vector<double> & Out,const std::vector<double> & In);

          cER_OneIm (const cER_OneIm &); // N.I.
          cER_Global * mERG;
          int         mKIm;
          Pt2di       mSz;
          std::string mName;
          bool        m2UseFCG;
          std::vector<double>               mCoeff;
          std::vector<cIncIntervale*>                   mII;
          std::vector<cER_SolOneCh *>                   mSols;

          bool   mL1Init;
          double mL1K0;
          double mL1Kx;
          double mL1Ky;
          double mSeuilPL1;
          int    mNbMesValidL1;

};

class cER_ParamOneSys
{
    public :
       cER_ParamOneSys(const std::vector<Pt2di> & DegSup);
       int NbVarTot() const;

       Pt2di  DegXYOfDegRadiom(int aDeg) const;
       int  DegMaxRadiom() const;
       void  write(ELISE_fp &) const;
       static cER_ParamOneSys  read(ELISE_fp &);
       static int  NbVarOfDeg(Pt2di aDeg);
       
    private :
       static int  NbVarOfDegI(int);
        
        std::vector<Pt2di> mDeg;
        int              mNbVarTot;
};


class cER_SysResol
{
    public :
       cER_SysResol(const cER_ParamOneSys &,cER_Global &);
       ~cER_SysResol();
       void InitSys(const std::vector<int> &);
       void Reset();
       void AddBlocInc(cIncIntervale * anII);
       static void AddCoeff(double aV0,Pt2dr,Pt2di aDeg,std::vector<double> &);
       
       cGenSysSurResol * Sys();
       Im1D_REAL8        GetSol();
    private :
       cER_SysResol(const cER_SysResol &); // N.I. 

       const cER_ParamOneSys  &          mParam;
       cER_Global &                      mERG;
       std::vector<cIncIntervale *>      mBlocsIncAlloc;
       cManipOrdInc                      mMOI;



       L2SysSurResol *                   mL2Sys;
       cGenSysSurResol *                 mSys;
};


class cElemGrapheIm
{
     public :
          cElemGrapheIm();
          void AddPt(const Pt2df & aP1,const Pt2df & aP2 );
          inline const int Nb() const;
          inline const Pt2df & PMin() const;
          inline const Pt2df & PMax() const;
          inline const  RMat_Inertie &  Inert1() const;
          inline bool  Ok() const;
           inline double & PdsEq(int aK);

          void CloseOK();

          void SetPackHom(const ElPackHomologue & aPckH);

          Pt2dr FromCNorm(const Pt2dr &);
          Pt2dr P1to2(const Pt2dr &);
          double  FactCorrec1to2(const Pt2dr &);

          void  SetParamL1(double aK0,double aKx,double aKy);


          
     private :
           bool           mOk;
           int            mNb;
           Pt2df          mPMin;
           Pt2df          mPMax;
           RMat_Inertie   mInert1;
           Pt2dr          mCdg1;
           Pt2dr          mUL1;
           Pt2dr          mUL2;
           double         mK0;
           double         mKx;
           double         mKy;
           cElHomographie mHom;
};

/*
   A faira dans cER_Global :

      End compute : (a call dans read)
      Verif end compute avant usage

*/

class cER_Global
{
       public :
           static cER_Global * Alloc
           (
               const std::vector<cER_ParamOneSys> &,
               const std::vector<cER_ParamOneSys> & ,
               Pt2di aSzIm                          ,
               const std::string &                aPatAdjustGlob,
               bool  ComputL1Cple
           );
           // DIM = nombre de cannaux
           cER_OneIm * AddIm(const std::string &,const Pt2di &);


           cER_MesureNIm & NewMesure2DPure();
           cER_MesureNIm & NewMesure2DGlob(const Pt2dr& );
           // void SuprLastMesure();
           ~cER_Global();
           static cER_Global * read(ELISE_fp &);
           static cER_Global * read(const std::string  &);

           const int & NbCh() const;
           const cER_ParamOneSys & ParamKCh(int aK) const;
           void  write(const std::string  &) const;
           void  write(ELISE_fp &) const;

           void Show1() const;
           int NBV() const;
           // bool In3D() const;
           int  NbIm() const;
           void SolveSys(double aPdsInit,double aPdsSingul,cER_Global * CompensRapGlob=0);
           double RadMoy(int aKCh ) const;
           bool UseImForCG(const std::string & aName) const;
           
           cER_OneIm *  ImG0();

           void CorrecVal(std::vector<double> &);
           inline double CorrecVal(double aV,int aKC);


           void Compute();
           void DoComputeL1Cple();
           double & PercCutAdjL1();
           bool   & Show();

           double DifL1Normal(const cER_MesureOneIm * aM1,const cER_MesureOneIm * aM2) const;
       private :
           bool ValideMesure(const cER_MesureOneIm&,const cER_MesureOneIm&);

           void DoComputeL1Cple(cER_OneIm * aI1,cER_OneIm * aI2,cElemGrapheIm &);
           void MakeStatL1ByIm(cER_OneIm *);
           void TestModelL1ByCple(cER_OneIm * aI1,cER_OneIm * aI2,cElemGrapheIm &);
           void AssertComputed();
           void AddStat(const cER_MesureNIm &);
           void OneItereSys(double aPdsInit,double aPdsSingul,cER_Global * CompensRG);

           cER_Global
           (
               const std::vector<cER_ParamOneSys> &,
               const std::vector<cER_ParamOneSys> & ,
               Pt2di aSzIm                        ,
               const std::string &                aPatAdjustGlob,
               bool                               isTop  ,
               bool                               isGlob  ,
               int                                ComputL1Cple

           );

           cER_Global(const cER_Global &) ; // N.I.
           void AddIm(cER_OneIm *);

           void InitSys();
           void OneItereOneChSys(int aNbCh,double aPdsInit,double aPdsSingul,cER_Global * CompensRapGlob);
           void AddOneObsTop(int aNbCh,double,const cER_MesureNIm &);
           void AddOneObsGlob(int aNbCh,double,const cER_MesureNIm &,cER_Global * CompensRapGlob);
           void PrevSingul(cER_OneIm * anIm,int aKCh,double aPds);



           cElRegex *                        mPatSelAdjGlob;
           std::string                       mNamePSAG;
           cER_Global*                       mErgTop;
           cER_Global*                       mErgG;
           bool                              mIsTop;
           bool                              mIsGlob;
           Pt2di                             mSzG;
           int                               mNumV;
           std::vector<cER_ParamOneSys>      mParam;
           std::vector<cER_ParamOneSys>      mParamGlob;
           int                               mNbCh;
           cER_OneIm *                      mImG0;
           std::vector<cER_SysResol *>    mSys;
           std::vector<double>              mMoy;
           std::vector<double>              mNbEch;


           std::map<std::string,cER_OneIm *> mDicoIm;
           std::vector<cER_OneIm *>          mVecIm;
           std::list<cER_MesureNIm>          mMes;
           cAMD_Interf     *                 mAMD;
           std::vector<int>                  mRnk;

           std::vector<double> mSomStat;
           std::vector<double> mMoyAv;
           std::vector<double> mMoyApr;
           std::vector<double> mMoyGrad;

           bool  mComputed;
           int  mComputL1Cple;
           double  mPercCutAdjL1;
           bool    mShow;

           // Par convention on choisit toujours P.x < P.y 
           std::map<Pt2di,cElemGrapheIm> mGrIm;
           int                           mNbCplOk;
           double                        mPdsTot;
           bool                          mGotOnePbL1;

};




#endif // _ELISE_TRAITEMENT_RADIOM_


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
