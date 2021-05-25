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



cParamModifGeomMTDNuage::cParamModifGeomMTDNuage
(
            double aScale,
            Box2dr aBox  ,
            bool   aDequant
) :
   mScale   (aScale),
   mBox     (aBox),
   mDequant (aDequant)
{
}



/***********************************************/
/*                                             */
/*       cElNuage3DMaille_FromImProf           */
/*                                             */
/***********************************************/


template <class Type,class TBase> 
        class cElNuage3DMaille_FromImProf : public cElNuage3DMaille
{
     public :

        ~cElNuage3DMaille_FromImProf() {}


         cElNuage3DMaille_FromImProf
         (
              const std::string &             aNameFile,
              const std::string &             aDir,
              const cXML_ParamNuage3DMaille & aNuage,
              Fonc_Num  aFMasq,
              Fonc_Num  aFProf,
              bool   WithEmptyData,
              bool   Dequant 
         );


         bool HasProfondeur() const { return true; }


         Im2DGen *   ImProf() const
         {
             return & const_cast<cElNuage3DMaille_FromImProf *>(this)->mIm;
         }
         double ProfOfIndex(const tIndex2D & anI) const
         {
             return PTab2PReel(mTIm.get(anI));
         }

         double ProfOfIndexInterpol(const Pt2dr & aP) const
         {
             return PTab2PReel(mTIm.getr(aP));
         }

         double   ProfOfIndexInterpolWithDef(const Pt2dr  & aP,double aDef) const
         {
             return PTab2PReel(mTIm.getr(aP,ElStdTypeScal<TBase>::RtoT(aDef)));
         }

         void SetProfOfIndex(const tIndex2D & anI,double aProf) 
         {
              mTIm.oset
              (
                    anI,
                    ElStdTypeScal<TBase>::RtoT(PReel2Tab(aProf))
              );
         }

         bool SetProfOfIndexIfSup(const tIndex2D & anI,double aProf)
         {
              TBase aV = ElStdTypeScal<TBase>::RtoT(PReel2Tab(aProf));
              if (aV > mTIm.get(anI))
              {
                  mTIm.oset(anI,aV);
                 return true;
              }
              return false;
         }


         void VerifParams() const;
         Pt3dr Loc_PtOfIndex(const tIndex2D & aP) const;

          void V_Save(const std::string & aNameP);

         Pt3dr Loc_IndexAndProfPixel2Euclid(const Pt2dr & anI,const double & anInvProf) const
         {

             return Loc_IndexAndProf2Euclid(anI,this->PTab2PReel(anInvProf));
         }

         Pt3dr Loc_Euclid2ProfPixelAndIndex(const   Pt3dr & aP) const
         {
              Pt3dr aRes = Loc_Euclid2ProfAndIndex(aP);
              return Pt3dr(aRes.x,aRes.y,this->PReel2Tab(aRes.z));
         }

         double  ProfEnPixel(const Pt2di & aP) const
         {
              return this->mTIm.getproj(aP);
         }
         double  ProfInterpEnPixel(const Pt2dr & aP) const 
         {
              return this->mTIm.getprojR(aP);
         }

        void ProfBouchePPV();

     protected :

         std::string       mNameFile;
         Im2D<Type,TBase>  mIm;
         TIm2D<Type,TBase> mTIm;
         double            mProf0;
         double            mResolProf;
         bool              mDequant;
        
     private :
         double PTab2PReel(const double aProf) const
         {
             return  mProf0 + mResolProf * aProf;
         }
         double PReel2Tab(const double aProf) const
         {
             return  (aProf-mProf0)/mResolProf;
         }
};

Fonc_Num sobel(Fonc_Num);

template <class Type,class TBase>  void  cElNuage3DMaille_FromImProf<Type,TBase>::ProfBouchePPV()
{
   Im2D<Type,TBase> aIPPV = BouchePPV(mIm,ImDef().in());


   int aNbTest = 7;
   for (int aK=0 ; aK< (aNbTest+2) ; aK++)
   {
       Symb_FNum aFMasq = ImDef().in();
       int aSzV = ElMax(1,ElSquare(aNbTest-aK));

       Fonc_Num aFLisse = rect_som(aIPPV.in_proj(),aSzV) /  ElSquare(1+2*aSzV);
       aFLisse  = aFLisse*(! aFMasq) + mIm.in() * aFMasq;
       ELISE_COPY(aIPPV.all_pts(),aFLisse,aIPPV.out());


   }
   ELISE_COPY(aIPPV.all_pts(),aIPPV.in(),mIm.out());
   ELISE_COPY(ImDef().all_pts(),1,ImDef().out());
}

template <class Type,class TBase>  void  cElNuage3DMaille_FromImProf<Type,TBase>::VerifParams() const
{
    const cXML_ParamNuage3DMaille&  aNuage =  this->Params(); 

    for
    (
          std::list<cVerifNuage>::const_iterator itVN=aNuage.VerifNuage().begin();
          itVN != aNuage.VerifNuage().end();
          itVN++
    )
    {
          Pt3dr aP1 = itVN->PointEuclid();
          Pt3dr aP2 = Loc_IndexAndProf2Euclid(itVN->IndIm(),PTab2PReel(itVN->Profondeur()));

          double aDist =  euclid(aP1-aP2);
          if (aDist>=aNuage.TolVerifNuage().Val())
          {
               cElWarning::ToVerifNuage.AddWarn
               (
                      "Dist = " + ToString(aDist) + " " + ToString(aP1.x) + " " +ToString(aP1.y) ,
                      __LINE__,
                     __FILE__
               );
          }
    }
}
  

template <class Type,class TBase>  
         Pt3dr cElNuage3DMaille_FromImProf<Type,TBase>::Loc_PtOfIndex(const tIndex2D & anI) const
{
    return Loc_IndexAndProf2Euclid(Pt2dr(anI),ProfOfIndex(anI));
}

template <class Type,class TBase> 
    void cElNuage3DMaille_FromImProf<Type,TBase>::V_Save(const std::string & aNameP)
{
    Tiff_Im::CreateFromIm(mIm,mDir+aNameP);
}

template <class Type,class TBase>  cElNuage3DMaille_FromImProf<Type,TBase>::cElNuage3DMaille_FromImProf
(
     const std::string &             aNameFile,
     const std::string &             aDir,
     const cXML_ParamNuage3DMaille & aNuage,
     Fonc_Num  aFMasq,
     Fonc_Num  aFProf,
     bool      WithEmpyData,
     bool      aDequant

)  :
   cElNuage3DMaille(aDir,aNuage,aFMasq,aNameFile,WithEmpyData), 
   mIm        (mSzData.x,mSzData.y),
   mTIm       (mIm),
   mProf0     (aNuage.Image_Profondeur().Val().OrigineAlti()),
   mResolProf (aNuage.Image_Profondeur().Val().ResolutionAlti()),
   mDequant   (aDequant)
{
    if (mDequant && (!WithEmpyData))
    {
        ElImplemDequantifier aDeq(mSzData);
        aDeq.DoDequantif(mSzData,aFProf);
        ELISE_COPY(mIm.all_pts(),aDeq.ImDeqReelle(),mIm.out());

    }
    else
    {
       ELISE_COPY ( mIm.all_pts(), aFProf, mIm.out());
    }
}

template class cElNuage3DMaille_FromImProf<INT2,INT>;
template class cElNuage3DMaille_FromImProf<float,double>;

/***********************************************/
/*                                             */
/*             cElN3D_EpipGen                  */
/*                                             */
/***********************************************/


/***********************************************/
/*                                             */
/*             cElN3D_EpipGen                  */
/*                                             */
/***********************************************/

   // Classe specialise dans le plus courant des nuages, celui
   // la geometrie epip generalisee (geometrie image et dyn en
   // 1/profondeur de champ)

template <class Type,class TBase> 
        class cElN3D_EpipGen : public cElNuage3DMaille_FromImProf<Type,TBase>
{
     public :

         cElN3D_EpipGen
         (
              const std::string &             aNameFile,
              const std::string &             aDir,
              const cXML_ParamNuage3DMaille & aNuage,
              Fonc_Num  aFMasq,
              Fonc_Num  aFProf,
              bool      aFaiscAndZ,
              bool      WithEmptyData,
              bool      Dequant
         );


         typedef cElNuage3DMaille::tIndex2D tIndex2D;

         Pt3dr Loc_IndexAndProf2Euclid(const   Pt2dr &,const double &) const;
         Pt3dr Loc_Euclid2ProfAndIndex(const   Pt3dr &) const;
         void  V_SetPtOfIndex(const tIndex2D & anI,const Pt3dr & aP3);

         cElNuage3DMaille * Clone() const;
         cElN3D_EpipGen<Type,TBase> * TypedClone() const;


         cElNuage3DMaille * V_ReScale
                                    (
                                        const Box2dr &Box,
                                        double aScale,
                                        const cXML_ParamNuage3DMaille &,
                                        Im2D_REAL4 anImPds,
                                        std::vector<Im2DGen*> aVNew,
                                        std::vector<Im2DGen*> aVOld
                                    ) ;

          double   ProfEuclidOfIndex(const tIndex2D & anI) const
          {
               return CorZInv(this->ProfOfIndex(anI));
          }
          void SetProfEuclidOfIndex(const tIndex2D & anI,double aProf)
          {
               this->SetProfOfIndex(anI,CorZInv(aProf));
          }


     private :
        inline double CorZInv(const double & aZ) const
        {
           return mZIsInv ? 1/aZ : aZ;
        }

        double  ProfOfPtE(const Pt3dr & aPe) const
        {
            if (mProfIsZ) return aPe.z;

            return  CorZInv(mIsSpherik?euclid(aPe-mCentre):scal(aPe-mCentre,mDirPl));
        }

        Pt3dr        mCentre;
        Pt3dr        mDirPl;
        bool         mProfIsZ;
        bool         mZIsInv;
        bool         mIsSpherik;
        CamStenope*  mCS;
        double       mProfC;

        ~cElN3D_EpipGen ()
        {
        }
};


template <class Type,class TBase> cElN3D_EpipGen<Type,TBase> *  cElN3D_EpipGen<Type,TBase>::TypedClone() const
{
    return new cElN3D_EpipGen<Type,TBase>
               (
                     this->NameFile(),
                     this->mDir,
                     this->mParams,
                     Fonc_Num(0),
                     Fonc_Num(El_CTypeTraits<Type>::TronqueR(-1e5)),
                     mProfIsZ,
                     this->mEmptyData,
                     false
                     // const_cast<cElN3D_EpipGen<Type,TBase> *>(this)->mImDef.in(),
                     // const_cast<cElN3D_EpipGen<Type,TBase> *>(this)->mIm.in()
               );
}

template <class Type,class TBase> cElNuage3DMaille *  cElN3D_EpipGen<Type,TBase>::Clone() const
{
   return TypedClone();
}


template <class Type,class TBase>  cElN3D_EpipGen<Type,TBase>::cElN3D_EpipGen
(
        const std::string &             aNameFile,
        const std::string &             aDir,
        const cXML_ParamNuage3DMaille & aNuage,
        Fonc_Num  aFMasq,
        Fonc_Num  aFProf,
        bool      aProfIsZ,
        bool      WithEmptyData,
        bool      Dequant

)  :
   cElNuage3DMaille_FromImProf<Type,TBase>(aNameFile,aDir,aNuage,aFMasq,aFProf,WithEmptyData,Dequant), 
   mProfIsZ   (aProfIsZ)
{
	mCentre	   = this->mCam->OrigineProf();
	mDirPl	   = this->Params().DirFaisceaux();
	mZIsInv    = this->Params().ZIsInverse();
	mIsSpherik = this->Params().IsSpherik().Val();
	mCS        = (mIsSpherik ? this->mCam->DownCastCS() : 0);
	mProfC     = scal(mDirPl,mCentre);
}

/*
extern Im2D_REAL4 ReduceImageProf(double aDifStd,Im2D_Bits<1>,Im2D_REAL4 aImProf, const Box2dr &aBox,double aScale,Im2D_REAL4 aImPds,std::vector<Im2DGen*>  aVNew,std::vector<Im2DGen*> aVOld);
extern Im2D_REAL4 ReduceImageProf(double aDifStd,Im2D_Bits<1>,Im2D_INT2 aImProf,const Box2dr &aBox,double aScale,Im2D_REAL4 aImPds,std::vector<Im2DGen*>  aVNew,std::vector<Im2DGen*> aVOld);
*/



template <class Type,class TBase> cElNuage3DMaille * cElN3D_EpipGen<Type,TBase>::V_ReScale
                                    (
                                        const Box2dr &aBox,
                                        double aScale,
                                        const cXML_ParamNuage3DMaille & aNewParam,
                                        Im2D_REAL4 anImPds,
                                        std::vector<Im2DGen*> aVNewAttr,
                                        std::vector<Im2DGen*> aVOldAttr
                                    ) 
{
    // RatioResolAltiPlani();

    // Im2DGen * anOld = this->mAttrs[0]->Im();


   // Fonc_Num aFoncProf = this->ReScaleAndClip(this->mImDef.in(0)*this->mIm.in(0),aBox._p0,aScale)/Max(1e-5,anImPds.in());

    /*
       for (int aKA=0 ; aKA<int(mAttrs.size()) ; aKA++)
       {
            Im2DGen * anOld = this->mAttrs[aKA]->Im();
            Im2DGen * aNew = anOld->ImOfSameType(Pt2di(aSz));

            aRes->mAttrs.push_back(new cLayerNuage3DM(aNew,mAttrs[aKA]->Name()));
        }
        aRes->mGrpAttr = mGrpAttr;
    */

   double aDifStd = 0.5;
   if (aNewParam.RatioResolAltiPlani().IsInit() && (aNewParam.Image_Profondeur().IsInit()))
   {
        ElAffin2D aAfM2C = Xml2EL(this->mParams.Orientation().OrIntImaM2C());
        double aResol = (euclid(aAfM2C.I10()) + euclid(aAfM2C.I01()))/2.0;

        aDifStd  = (1/aResol) * (1/this->mParams.Image_Profondeur().Val().ResolutionAlti())   * (this->mParams.RatioResolAltiPlani().Val()) ;
        aDifStd *= 0.5;
   }

   Im2D_REAL4 aRedProf = ReduceImageProf(aDifStd,this->mImDef,this->mIm,aBox,aScale,anImPds,aVNewAttr,aVOldAttr);
   Fonc_Num  aFoncProf = aRedProf.in();



   cElN3D_EpipGen<float,double> * aRes = new cElN3D_EpipGen<float,double>
                                             (
                                                 this->NameFile(),
                                                 this->mDir,
                                                 aNewParam,
                                                 anImPds.in(0) > 0.1,
                                                 aFoncProf,
                                                 //   this->ReScaleAndClip(this->mImDef.in(0)*this->mIm.in(0),aBox._p0,aScale)/Max(1e-5,anImPds.in()),
                                                 mProfIsZ,
                                                 this->mEmptyData,
                                                 false
                                             );

   return aRes;
}


template <class Type,class TBase>  
         Pt3dr cElN3D_EpipGen<Type,TBase>::Loc_IndexAndProf2Euclid(const Pt2dr & anI,const double & anInvProf) const
{
if (MPD_MM())
{
    // std::cout << "TTTttttttttttt " << mProfIsZ  << " " << anI << " " << anInvProf << "\n";
}

  if (mProfIsZ) 
  {
       // std::cout << anInvProf << " " << anI << " "  << this->mCam->F2AndZtoR3(anI,anInvProf)  << "\n";
      return  this->mCam->ImEtZ2Terrain(anI,anInvProf);
  }

   double aProf =   CorZInv(anInvProf);


   if (mIsSpherik)
   {
       return mCS->ImEtProfSpherik2Terrain(anI,aProf);
   }


    ElSeg3D aSeg =  this->mCam->Capteur2RayTer(anI) ;
    Pt3dr aRay =aSeg.Tgt();
    Pt3dr aC  = aSeg.P0();

   return   aC
          + aRay * (aProf/scal(aRay,mDirPl));
}


template <class Type,class TBase>
  Pt3dr cElN3D_EpipGen<Type,TBase>::Loc_Euclid2ProfAndIndex(const   Pt3dr & aPe) const
{
    Pt2dr aPIm =  this->mCam->Ter2Capteur(aPe);
    return Pt3dr(aPIm.x,aPIm.y,ProfOfPtE(aPe));
/*
    double aProf =   mIsSpherik                             ?
                     euclid(aPe-mCentre)                    :
                     scal(aPe-mCentre,mDirPl)               ;
    Pt2dr aPIm =  this->mCam->R3toF2(aPe);
    return Pt3dr(aPIm.x,aPIm.y,CorZInv(aProf));
*/
}




template <class Type,class TBase>  
         void cElN3D_EpipGen<Type,TBase>::V_SetPtOfIndex(const tIndex2D & anI,const Pt3dr & aP3)
{
   this->SetProfOfIndex(anI,ProfOfPtE(aP3));
/*
   double aProf =    mIsSpherik                             ?
                     euclid(aP3-mCentre)                    :
                     scal(mDirPl,aP3-mCentre)               ;
   this->SetProfOfIndex(anI, CorZInv(aProf));
*/
}


template class cElN3D_EpipGen<INT2,INT>;
template class cElN3D_EpipGen<float,double>;

/***********************************************/
/*                                             */
/*             cElNuage3DMaille                */
/*                                             */
/***********************************************/


cElNuage3DMaille * cElNuage3DMaille::FromParam
                   (
                       const std::string & aNameFile,
                       const cXML_ParamNuage3DMaille & aParamOri,
                       const std::string & aDir,
                       const std::string & aMasqSpec,
                       double ExagZ,
                       const cParamModifGeomMTDNuage * aPMG,
                       bool  WithEmptyData
                   )
{
  cXML_ParamNuage3DMaille aParam = aParamOri;
  Box2di aBox(Pt2di(0,0),aParam.NbPixel());
  bool Dequant = false;

  if (aPMG)
  {
       double aScale = aPMG->mScale;
       if (aScale != 1.0)
       {
           std::string aMes = "Scale=" + ToString(aScale);
           cElWarning::ScaleInNuageFromP.AddWarn(aMes,__LINE__,__FILE__);
       }
       Pt2di  aP0 = round_down(aPMG->mBox._p0/aScale);
       Pt2di  aP1 = round_up(aPMG->mBox._p1/aScale);

       aBox = Inf(aBox,Box2di(aP0,aP1));

       aParam.NbPixel() = aBox.sz();

       ElAffin2D aAfM2C = Xml2EL(aParam.Orientation().OrIntImaM2C()); // RPCNuage
       
       aAfM2C   =   ElAffin2D::trans(-Pt2dr(aBox._p0)) * aAfM2C;
       // aParam.Orientation().Val().OrIntImaM2C()= El2Xml(aAfM2C); // RPCNuage
       aParam.Orientation().OrIntImaM2C().SetVal(El2Xml(aAfM2C));
       Dequant = aPMG->mDequant;
  }


  std::string aMasq =  aDir+aParam.Image_Profondeur().Val().Masq();
  if (aMasqSpec!="")
         aMasq = aMasqSpec;

  GenIm::type_el aTypeEl = GenIm::real4;
  Fonc_Num aFMasq = 0;
  Fonc_Num aFProf = 1;
  if (! WithEmptyData)
  {
     aFMasq =   trans(Tiff_Im::BasicConvStd(aMasq).in(0),aBox._p0);
     Tiff_Im aTP = Tiff_Im::BasicConvStd(aDir+aParam.Image_Profondeur().Val().Image());
     aFProf =  trans(aTP.in_proj()*ExagZ,aBox._p0);
     aTypeEl = aTP.type_el();
  }


   if (aParam.Image_Profondeur().IsInit())
   {
      bool aFaiscClassik =  (   (aParam.Image_Profondeur().Val().GeomRestit()==eGeomMNTFaisceauIm1PrCh_Px1D)
                             || (aParam.Image_Profondeur().Val().GeomRestit()==eGeomMNTFaisceauIm1PrCh_Px2D)
                             || (aParam.Image_Profondeur().Val().GeomRestit()==eGeomMNTEuclid)
                             || (aParam.Image_Profondeur().Val().GeomRestit()==eGeomMNTFaisceauPrChSpherik)
                            );

      bool aProfIsZ =  (      (aParam.Image_Profondeur().Val().GeomRestit()==eGeomMNTFaisceauIm1ZTerrain_Px1D)
                             || (aParam.Image_Profondeur().Val().GeomRestit()==eGeomMNTFaisceauIm1ZTerrain_Px2D)
                         );


       if (aFaiscClassik || aProfIsZ)
       {
           switch (aTypeEl)
           {
               case GenIm::int2 :
                    if (Dequant) 
                        return new cElN3D_EpipGen<float,double>(aNameFile,aDir,aParam,aFMasq,aFProf,aProfIsZ,WithEmptyData,true);
                    else
                        return new cElN3D_EpipGen<INT2,INT>(aNameFile,aDir,aParam,aFMasq,aFProf,aProfIsZ,WithEmptyData,false);
               break;
               case GenIm::real4 :
                    return new cElN3D_EpipGen<float,double>(aNameFile,aDir,aParam,aFMasq,aFProf,aProfIsZ,WithEmptyData,false);
               break;

               default :
                   std::cout << "NAME " << aDir+aParam.Image_Profondeur().Val().Image() << "\n";
                   ELISE_ASSERT(false,"Type Image non gere dans cElNuage3DMaille::FromParam");
               break;
           }
       }
   }

   ELISE_ASSERT(false,"cElNuage3DMaille::FromParam");
   return 0;
}

cElNuage3DMaille * cElNuage3DMaille::FromFileIm(const std::string & aFile)
{
   return FromFileIm(aFile,"XML_ParamNuage3DMaille","",1.0);
}

cElNuage3DMaille * cElNuage3DMaille::FromFileIm
                   (
                          const std::string & aFile,
                          const std::string & aTag,
                          const std::string & aMasq,
                          double ExagZ
                   )
{
   std::string aDir,aNF;
   SplitDirAndFile(aDir,aNF,aFile);
   return FromParam
          (
                aFile,
                StdGetObjFromFile<cXML_ParamNuage3DMaille>
                (
                      aFile,
                      StdGetFileXMLSpec("SuperposImage.xml"),
                      aTag,
                      "XML_ParamNuage3DMaille"
                ),
                aDir,
                aMasq,
                ExagZ
         );
}


cXML_ParamNuage3DMaille XML_Nuage(const std::string & aName)
{
    return StdGetObjFromFile<cXML_ParamNuage3DMaille>
           (
                aName,
                StdGetFileXMLSpec("SuperposImage.xml"),
                "XML_ParamNuage3DMaille",
                "XML_ParamNuage3DMaille"
           );
}


Fonc_Num Pix2Z(const cXML_ParamNuage3DMaille & aCloud,Fonc_Num aF)
{
   const cImage_Profondeur & aIP = aCloud.Image_Profondeur().Val();
   return aIP.OrigineAlti() + aIP.ResolutionAlti() * aF ;
}

Fonc_Num Z2Pix(const cXML_ParamNuage3DMaille & aCloud,Fonc_Num aF)
{
   const cImage_Profondeur & aIP = aCloud.Image_Profondeur().Val();
   return (aF-aIP.OrigineAlti()) /  aIP.ResolutionAlti()  ;
}

Fonc_Num Pix2Pix(const cXML_ParamNuage3DMaille &Out,Fonc_Num aF,const cXML_ParamNuage3DMaille & In)
{
   return Z2Pix(Out,Pix2Z(In,aF));
}

Fonc_Num Pix2Pix
         (
             const cXML_ParamNuage3DMaille &Out,
             const cXML_ParamNuage3DMaille & In,
             const std::string & aDir
         )
{
   const cImage_Profondeur & aIP = In.Image_Profondeur().Val();
   return Pix2Pix
          (
             Out,
             Tiff_Im::StdConvGen(aDir+aIP.Image(),1,true,false).in(),
             In
          );
}
/*
*/

cElNuage3DMaille * NuageWithoutDataWithModel(const std::string & aName,const std::string & aModel)
{
   cXML_ParamNuage3DMaille aParam =   XML_Nuage(aName);

   if (aModel!="")
   {
       cXML_ParamNuage3DMaille aPMod =   XML_Nuage(aModel);
       aParam.Image_Profondeur().Val().OrigineAlti() = aPMod.Image_Profondeur().Val().OrigineAlti();
       aParam.Image_Profondeur().Val().ResolutionAlti() = aPMod.Image_Profondeur().Val().ResolutionAlti();
   }

   return cElNuage3DMaille::FromParam
           (
                 aName,
                 aParam,
                 DirOfFile(aName),
                 "",
                 1.0,
                 (cParamModifGeomMTDNuage *) 0,
                 true
           );
}

cElNuage3DMaille * NuageWithoutData(const cXML_ParamNuage3DMaille & aParam,const std::string & aName)
{
   return cElNuage3DMaille::FromParam
           (
                 aName,
                 aParam,
                 DirOfFile(aName),
                 "",
                 1.0,
                 (cParamModifGeomMTDNuage *) 0,
                 true
           );
}


cElNuage3DMaille * NuageWithoutData(const std::string & aName)
{
    return NuageWithoutDataWithModel(aName,"");
}

bool GeomCompatForte(cElNuage3DMaille * aN1,cElNuage3DMaille *aN2)
{
    if (aN1->SzGeom() != aN2->SzGeom()) return false;

    Box2dr aBox (Pt2dr(1,1),Pt2dr(aN1->SzGeom())-Pt2dr(1,1));

    for (int aK = 0 ; aK< 10 ; aK++)
    {
         Pt2dr aP = aBox.RandomlyGenereInside();

         Pt3dr aQ3_1 = aN1->Loc_IndexAndProfPixel2Euclid(aP,10.0);
         Pt3dr aQ3_2 = aN2->Loc_IndexAndProfPixel2Euclid(aP,10.0);

         if (euclid(aQ3_1-aQ3_2) > 1e-5) return false;
    }

    return true;
}


cRawNuage::cRawNuage(Pt2di aSz) :
   mImX   (aSz.x,aSz.y),
   mTX     (mImX),
   mImY   (aSz.x,aSz.y),
   mTY     (mImY),
   mImZ   (aSz.x,aSz.y),
   mTZ     (mImZ)
{
}

Im2D_REAL4 cRawNuage::ImX() {return mImX;}
Im2D_REAL4 cRawNuage::ImY() {return mImY;}
Im2D_REAL4 cRawNuage::ImZ() {return mImZ;}

void cRawNuage::SetPt(const  Pt2di & anI,const Pt3dr & aP)
{
   mTX.oset(anI,aP.x);
   mTY.oset(anI,aP.y);
   mTZ.oset(anI,aP.z);
}

Pt3dr  cRawNuage::GetPt(const Pt2di & anI) const
{
   return Pt3dr
          (
              mTX.get(anI),
              mTY.get(anI),
              mTZ.get(anI)
          );
}

cRawNuage   cElNuage3DMaille::GetRaw() const
{
   Pt2di aSz = SzUnique();
   cRawNuage aRes(aSz);

   Pt2di aP0;
   for (aP0.x=0 ; aP0.x<aSz.x ; aP0.x++)
   {
       for (aP0.y=0 ; aP0.y<aSz.y ; aP0.y++)
       {
            if (IndexHasContenu(aP0))
            {
                aRes.SetPt(aP0,PtOfIndex(aP0));
            }
            else
            {
                aRes.SetPt(aP0,Pt3dr(0,0,0));
            }
       }
   }

   return aRes;
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
