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

#ifndef _BUNDLEGEN_H_
#define _BUNDLEGEN_H_


class cBGC3_Modif2D ; //   : public cBasicGeomCap3D
class cPolynomial_BGC3M2D ;//  : public cBGC3_Modif2D
class cPolynBGC3M2D_Formelle; // : public cGenPDVFormelle
class cOneEq_PBGC3M2DF;


/*

   N ((i,j)+D(i,j)) = R N(i,j) = (I + W e ^ N(i,j))
  
   (Id + J * D(i,j) )  = I + W e ^ N(i,j))

    D(i,j) = J-1 W ^ N(i,j)

*/

class cBGC3_Modif2D  : public cBasicGeomCap3D
{
      public : 
           cBGC3_Modif2D(cBasicGeomCap3D * aCam0,const std::string & aName,const std::string &aNameIma);




           virtual ElSeg3D  Capteur2RayTer(const Pt2dr & aP) const ;
           virtual Pt2dr    Ter2Capteur   (const Pt3dr & aP) const ;
           Pt2dr            Ter2CapteurSsCorrec   (const Pt3dr & aP) const ;
           virtual Pt2di    SzBasicCapt3D() const ;
           virtual double ResolSolOfPt(const Pt3dr &) const ;
           virtual bool  CaptHasData(const Pt2dr &) const ;
           virtual bool     PIsVisibleInImage   (const Pt3dr & aP,cArgOptionalPIsVisibleInImage  * = 0) const ;
           virtual Pt3dr RoughCapteur2Terrain   (const Pt2dr & aP) const ;
           virtual double GetVeryRoughInterProf() const;

  // Optical center 
           virtual bool     HasOpticalCenterOfPixel() const; // 1 - They are not alway defined
  // When they are, they may vary, as with push-broom, Def fatal erreur (=> Ortho cam)
           virtual Pt3dr    OpticalCenterOfPixel(const Pt2dr & aP) const ;

           virtual inline Pt2dr CamInit2CurIm(const Pt2dr & aP) const{return aP+DeltaCamInit2CurIm(aP);}
           virtual inline Pt2dr CurIm2CamInit(const Pt2dr & aP) const{return aP+DeltaCurIm2CamInit(aP);}

           cBasicGeomCap3D * CamSsCor();

            virtual Pt2dr DeltaCamInit2CurIm(const Pt2dr & aP) const = 0;
            virtual Pt2dr DeltaCurIm2CamInit(const Pt2dr & aP) const ;
            Pt2dr ImRef2Capteur   (const Pt2dr & aP) ;
            double ResolImRefFromCapteur() const;


           Pt3dr ImEtProf2Terrain(const Pt2dr & aP,double aZ) const;
           Pt3dr ImEtZ2Terrain(const Pt2dr & aP,double aZ) const;

           double GetAltiSol() const ;
           Pt2dr GetAltiSolMinMax() const ;
           bool AltisSolIsDef() const ;
           bool AltisSolMinMaxIsDef() const;
           bool IsRPC() const;
           const std::string & NameIma() const;


      protected  : 
            cBasicGeomCap3D * mCam0;
            std::string       mNameFileCam0;
            std::string       mNameIma;
            Pt2di  mSz;

      private  : 

            // Ter2Cam (x) = x + DifCorTer2Cal(x)

};


class cPolynomial_BGC3M2D  : public cBGC3_Modif2D
{
      public : 
           cPolynomial_BGC3M2D(const cSystemeCoord * aChSys,cBasicGeomCap3D * aCam0,const std::string & aName,const std::string & aNameIma,int aDegree,double aRandomPert=0);
           Pt2dr DeltaCamInit2CurIm(const Pt2dr & aP) const ;
           inline Pt2dr ToPNorm(const Pt2dr aP) const {return (aP-mCenter)/mAmpl;}
           inline Pt2dr FromPNorm(const Pt2dr aP) const {return aP*mAmpl + mCenter;}

           std::vector<double> & Cx();
           std::vector<double> & Cy();
           inline int DegX(int aK) const {return mDegX.at(aK);}
           inline int DegY(int aK) const {return mDegY.at(aK);}
           inline const int & DegreMax()   const {return mDegreMax;}
           inline const double  &       Ampl() const {return mAmpl;}
           inline const Pt2dr &      Center() const {return mCenter;}
  //         Pt3dr RTLCenter() const;

	   void Show() const;

           virtual std::string Save2XmlStdMMName(  cInterfChantierNameManipulateur * anICNM,
                                        const std::string & aOriOut,
                                        const std::string & aNameImClip,
                                        const ElAffin2D & anOrIntInit2Cur
                    ) const;

           std::string DirSave(const std::string & aDirLoc,const std::string & aPref,bool Create=true) const;
           std::string NameSave(const std::string & aDirLoc,const std::string & aPref) const;

           cXml_CamGenPolBundle ToXml() const;
           // WithAffine recupere eventuellement la deformation affine, si vaut 0 et def existe => erreur
           static  cPolynomial_BGC3M2D * NewFromFile(const std::string &,cBasicGeomCap3D **  WithAffine= 0);
      private : 
           void SetMonom(const cMonomXY & aMon,std::vector<double> &);
           void SetMonom(const std::vector<cMonomXY> & aMon,std::vector<double> &);

           void Show(const std::string & aMes,const std::vector<double> & aCoef) const;
           void ShowMonome(const std::string & , int aDeg) const;
           void SetPow(const Pt2dr & aPN) const;
 
           const cSystemeCoord *  mPtrChSys;
           
           int                 mDegreMax;
           Pt2dr               mCenter;
           double              mAmpl;
           std::vector<double> mCx;
           std::vector<double> mCy;

           static std::vector<int> mDegX;
           static std::vector<int> mDegY;

           cXml_PolynXY ExporOneCor(const std::vector<double> & aCoeff) const;


           static std::vector<double> mPowX;
           static std::vector<double> mPowY;
           mutable Pt2dr  mCurPPow;


};

class cOneEq_PBGC3M2DF : public cElemEqFormelle,
                         public cObjFormel2Destroy

{
    public :
       cOneEq_PBGC3M2DF(cPolynBGC3M2D_Formelle &,std::vector<double > &);

       Fonc_Num  EqFormProjCor(Pt2d<Fonc_Num> aP);
       
   private :
       std::vector<Fonc_Num>     mVFCoef;
       cPolynBGC3M2D_Formelle *  mPF;
       cPolynomial_BGC3M2D*      mCamCur;
};


class cCellPolBGC3M2DForm
{
      public :
          cCellPolBGC3M2DForm(Pt2dr mPt,cPolynBGC3M2D_Formelle * aPF,int aDim);
          cCellPolBGC3M2DForm();
          void InitRep(cPolynBGC3M2D_Formelle * aPF);
          void SetGrad(const Pt2dr & aGX,const Pt2dr & aGy);
      
          Pt2dr  ProjOfTurnMatr(bool & Ok,const ElMatrix<double> & Mat);


          cPolynBGC3M2D_Formelle * mPF;
          Pt2dr                    mPtIm;
          Pt3dr                    mNorm;
          Pt3dr                    mCenter;
          Pt3dr                    mPTer;
          bool                     mActive;
          bool                     mHasDep;
          int                      mDim;
          std::vector<Pt2dr>       mValDep;
};

class cPolynBGC3M2D_Formelle : public cGenPDVFormelle
{

    public  :


         cPolynBGC3M2D_Formelle  * ThisIsConstructeur();
         const cPolynBGC3M2D_Formelle  * ThisIsConstructeur() const;


         friend class cOneEq_PBGC3M2DF;
         friend class cCellPolBGC3M2DForm;

         cPolynBGC3M2D_Formelle(cSetEqFormelles & aSet,cPolynomial_BGC3M2D aCam0,bool GenCodeAppui,bool GenCodeAttach,bool   GenCodeRot);
         void GenerateCode(Pt2d<Fonc_Num>,const std::string &,cIncListInterv &);
         cIncListInterv & IntervAppuisPtsInc() ;
         void PostInit();
         const cBasicGeomCap3D * GPF_CurBGCap3D() const ;
         cBasicGeomCap3D * GPF_NC_CurBGCap3D() ;
         Pt2dr AddEqAppuisInc(const Pt2dr & aPIm,double aPds, cParamPtProj &,bool IsEqDroite,cParamCalcVarUnkEl*);
         
         const cPolynomial_BGC3M2D *  TypedCamCur() const { return & mCamCur; }
         cPolynomial_BGC3M2D *  TypedCamCur() { return & mCamCur; }
         void AddEqAttachGlob(double aPds,bool Cur,int aNbPts,CamStenope * aKnownSol);
         cBasicGeomCap3D *   CamSsCorr() const ;

         // cCellPolBGC3M2DForm & Cell(const Pt2di & aP) {return mVCells.at(aP.y).at(aP.x);}
         cCellPolBGC3M2DForm & Cell(const Pt2di & aP) {return mVCells[aP.y][aP.x];}
         const cCellPolBGC3M2DForm & Cell(const Pt2di & aP) const {return mVCells[aP.y][aP.x];}

         bool CellHasValue(const Pt2di &) const;
         bool CellHasGradValue(const Pt2di &) const;

         void TestRot(const Pt2di & aP0,const Pt2di &aP1,double & aSomD,double & aSomR,ElMatrix<double> *);
         Pt2di SzCell() {return Pt2di(mNbCellX,mNbCellY);}
         Pt2dr  P2dNL(const Pt2dr & aPt) const;

         void AddEqRot(const Pt2di & aP0,const Pt2di &aP1,double aPds);
         void AddEqRotGlob(double aPds);
         double ModifInTervGrad(const double & aVal,const double & aBorne) const;

    private :
         Pt2dr DepSimul(const Pt2dr & aP,const ElMatrix<double> & aMat);
         Pt2dr DepOfKnownSol(const Pt2dr & aP,CamStenope *);
         cPolynBGC3M2D_Formelle(const cPolynBGC3M2D_Formelle &); // N.I.


         Pt2dr   PtOfRot(const cCellPolBGC3M2DForm &,const ElMatrix<double> & aMat);
         

   // ==> To unvirtualize cGenPDVFormelle 
         Pt2d<Fonc_Num>  EqFormProj();
         Pt2d<Fonc_Num>  EqFixedVal();
         Pt2d<Fonc_Num>  EqAttachRot();


         Pt2d<Fonc_Num>  FormalCorrec(Pt2d<Fonc_Num> aPF,cVarEtat_PhgrF aFAmpl,cP2d_Etat_PhgrF aFCenter);


         void AddEqAttach(Pt2dr aPIm,double aPds,bool Cur,CamStenope * aKnownSol);

         cBasicGeomCap3D *   mCamSsCorr;
         cPolynomial_BGC3M2D mCamInit;
         cPolynomial_BGC3M2D mCamCur;



         cEqfP3dIncTmp * mEqP3I;

         cVarEtat_PhgrF    mFAmplAppui;
         cVarEtat_PhgrF    mFAmplFixVal;
         cVarEtat_PhgrF    mFAmplAttRot;
         cP2d_Etat_PhgrF   mFCentrAppui;
         cP2d_Etat_PhgrF   mFCentrFixVal;
         cP2d_Etat_PhgrF   mFCentrAttRot;

         cP3d_Etat_PhgrF   mFP3DInit;
         cP2d_Etat_PhgrF   mFProjInit;

         cP2d_Etat_PhgrF   mFGradX;
         cP2d_Etat_PhgrF   mFGradY;
         cP2d_Etat_PhgrF   mFGradZ;
         cP2d_Etat_PhgrF   mObsPix;


         cP2d_Etat_PhgrF   mPtFixVal;
         cP2d_Etat_PhgrF   mFixedVal;

         cP2d_Etat_PhgrF    mRotPt;
         cP2d_Etat_PhgrF    mDepR1;
         cP2d_Etat_PhgrF    mDepR2;
         cP2d_Etat_PhgrF    mDepR3;


         cOneEq_PBGC3M2DF    mCompX;
         cOneEq_PBGC3M2DF    mCompY;
         std::string         mNameType;
         std::string         mNameAttach;
         std::string         mNameRot;
         cIncListInterv      mLIntervResiduApp;
         cIncListInterv      mLIntervAttach;
         cIncListInterv      mLIntervRot;
         cElCompiledFonc *   mFoncEqResidu;
         cElCompiledFonc *   mFoncEqAttach;
         cElCompiledFonc *   mFoncEqRot;
         int                 mNbCellX;
         int                 mNbCellY;
         Pt2di               mIndCenter;

         std::vector<std::vector<cCellPolBGC3M2DForm> > mVCells;
         Pt3dr               mCenterGlob;
         ElRotation3D        mRotL2W;
         ElMatrix<double>    mMatW2Loc;

         static double                           mNbPixOfEpAngle;
         double                                  mEspilonAngle;
         std::vector<ElMatrix<double> >          mEpsRot;
         static double                           mEpsGrad;
         cSubstitueBlocIncTmp *                  mBufSubRot;
         int                                     mDimMvt;
};


#endif //  _BUNDLEGEN_H_



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
