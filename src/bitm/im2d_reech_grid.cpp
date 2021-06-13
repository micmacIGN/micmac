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





/**************************************************/
/*                                                */
/*                RImGrid                         */
/*                                                */
/**************************************************/

double AdaptPas(double aStep,double aSz)
{
  return aSz / ElMax(1,round_ni(aSz/aStep));
}

Pt2dr AdaptPas(Pt2dr aStep,Pt2dr aSz)
{
   return Pt2dr
          (
                AdaptPas(aStep.x,aSz.x),
                AdaptPas(aStep.y,aSz.y)
          );
}

double MessDeb(const Pt2di & aSz, Pt2dr aStep,Pt2dr aP0,Pt2dr aP1,bool Adapt,double aVal)
{
   std::cout << "SZGR " << aSz << "  Step" << aStep << " P0" << aP0 << aP1 << " Ada " << Adapt << "\n";
   return aVal;
}

RImGrid::RImGrid
(
     bool AdaptStep,
     Pt2dr aP0,
     Pt2dr aP1,
     Pt2dr  aStepGr,
     const std::string & aName,
     Pt2di aSz
) :
  mP0     (Inf(aP0,aP1)),
  mP1     (Sup(aP0,aP1)),
  mStepGr (AdaptStep ? AdaptPas(aStepGr,mP1-mP0) : aStepGr),
  mSzGrid ( (aSz!=Pt2di(0,0)) ? 
            aSz               :
            (
                AdaptStep                                        ?
	        (round_ni((aP1-aP0).dcbyc(mStepGr))+Pt2di(1,1))  :
                (round_up((aP1-aP0).dcbyc(mStepGr))+Pt2di(1,1))
            )
          ),
  mDef    (-1e20),
  // mDef    (MessDeb(mSzGrid,mStepGr,aP0,aP1,AdaptStep,-1e20)),
  mGrid   (mSzGrid.x,mSzGrid.y,mDef),
  mTim    (new TIm2D<REAL,REAL>(mGrid)),
  mName   (aName),
  mStepAdapted (AdaptStep)
{
}



RImGrid::RImGrid
(
     Pt2dr      anOrigine,
     Pt2dr      aStep,
     Im2D_REAL8 anIm
) :
   mP0      (anOrigine),
   mP1      (mP0+Pt2dr(anIm.sz() -Pt2di(1,1)).mcbyc(aStep)),
   mStepGr  (aStep),
   mSzGrid  (anIm.sz()),
   mDef     (-1e20),
   mGrid    (anIm),
   mTim     (new TIm2D<REAL,REAL>(mGrid)),
   mName    ("No Name"),
   mStepAdapted (true)
{
}

const Pt2dr & RImGrid::P0() const { return mP0; }
const Pt2dr & RImGrid::P1() const { return mP1; }
bool  RImGrid::StepAdapted() const {return mStepAdapted;}


const std::string & RImGrid::Name() const
{
	return mName;
}

void RImGrid::write(class  ELISE_fp & aFile) const
{
   ELISE_ASSERT(false,"No More Support for RImGrid::write");
/*
   aFile.write(mP0);
   aFile.write(mP1);
   aFile.write(mStepGr);
   aFile.write(mSzGrid);
   WritePtr(aFile,mGrid.tx()*mGrid.ty(),mGrid.data_lin());
   */
}

RImGrid * RImGrid::read(class  ELISE_fp & aFile) 
{
   ELISE_ASSERT(false,"No More Support for RImGrid::read");

   return 0;
/*
   Pt2dr aP0 = aFile.read(&aP0);
   Pt2dr aP1 = aFile.read(&aP1);
   REAL  aStep = aFile.read(&aStep);
   Pt2di aSZ;
   aSZ = aFile.read(&aSZ);
std::cout << aSZ << "\n";
   RImGrid * aRes   = new RImGrid(true,aP0,aP1,Pt2dr(aStep,aStep),"toto",aSZ);

   ELISE_ASSERT(aSZ==aRes->mSzGrid,"Incoherent Sz in RImGrid::read");

   Im2D_REAL8 aGrid = aRes->mGrid;
   ReadPtr(aFile,aGrid.tx()*aGrid.ty(),aGrid.data_lin());

   return aRes;
 */
}




void RImGrid::TranlateIn(Pt2dr aDP)
{
    mP0 += aDP;
    mP1 += aDP;
}

RImGrid::~RImGrid()
{
   delete mTim;
}

Pt2dr RImGrid::ToReal(Pt2dr aP) const
{
	return aP.mcbyc(mStepGr) +mP0;
}
Pt2dr RImGrid::ToGrid(Pt2dr aP) const
{
	return (aP-mP0).dcbyc(mStepGr);
}
Pt2di  RImGrid::SzGrid() const
{
   return mSzGrid;
}

REAL RImGrid::Value(Pt2dr  aP) const
{
   aP = ToGrid(aP);
   return mTim->getr
	  (
	       Pt2dr
	       (
                   ElMax(0.0,ElMin(aP.x,mSzGrid.x-1.001)),
                   ElMax(0.0,ElMin(aP.y,mSzGrid.y-1.001))
               )
	  );
}

REAL RImGrid::ValueAndDer(Pt2dr aP,Pt2dr & aDer)
{
   aP = ToGrid(aP);
   Pt3dr aP3 =  mTim->getVandDer
	        (
	             Pt2dr
	             (
                         ElMax(0.0,ElMin(aP.x,mSzGrid.x-1.001)),
                         ElMax(0.0,ElMin(aP.y,mSzGrid.y-1.001))
                     )
	        );
   aDer.x = aP3.x / mStepGr.x;
   aDer.y = aP3.y / mStepGr.y;

   return aP3.z;
}



void RImGrid::InitGlob( Fonc_Num aFonc )
{
     ELISE_COPY
     (
         mGrid.all_pts(),
         aFonc[Virgule(FX*mStepGr.x+mP0.x,FY*mStepGr.y+mP0.y)],
         mGrid.out()
     );
}

void RImGrid::SetValueGrid(Pt2di aP,REAL aV)
{
     ELISE_ASSERT
     (
        (aP.x>=0)&&(aP.x<mSzGrid.x)&&(aP.y>=0)&&(aP.y<mSzGrid.y),
	"Out Of RImGrid::SetValueGrid"
     );
     mGrid.data()[aP.y][aP.x] = aV;
}

void  RImGrid::ExtDef()
{
     Neighbourhood V4 = Neighbourhood::v4();
     Neighbourhood V8 = Neighbourhood::v8();
     Liste_Pts_INT2 anOldL(2);
     bool First = true;

     while(First || (!anOldL.empty()))
     {
         Flux_Pts aFlx = First ?
                         select(mGrid.all_pts(), mGrid.in()!=mDef):
		         anOldL.all_pts();

         Liste_Pts_INT2 aNewL(2);
         ELISE_COPY
         (
            dilate(aFlx,sel_func(V4,mGrid.in(mDef/2)==mDef)),
	    2*mDef,
            mGrid.out() | aNewL
         );
	 
         Neigh_Rel aROK = sel_func(V8,mGrid.in(mDef)!= mDef);
         ELISE_COPY
	 (
	     aNewL.all_pts(),
               aROK.red_sum( mGrid.in())
             / aROK.red_sum(1),
	     mGrid.out()
	 );

	 anOldL = aNewL;
	 First = false;
     }
}

void RImGrid::SetTrChScaleOut(REAL aChScale,REAL aTr)
{
   ELISE_COPY
   (
       mGrid.all_pts(),
       mGrid.in()*aChScale+aTr,
       mGrid.out()
   );
}

void RImGrid::SetTrChScaleIn(REAL aChScale,Pt2dr aTr)
{
   mStepGr = mStepGr * aChScale;
   mP0 = mP0*aChScale + aTr;
   mP1 = mP1*aChScale + aTr;
}


RImGrid *  RImGrid::NewChScale(REAL aChScale ,bool ModeMapping)
{
    RImGrid * aRes = new RImGrid
                         (
			     mStepAdapted,
                             mP0 * aChScale,
                             mP1 * aChScale,
                             mStepGr*aChScale
                         );

   ELISE_COPY
   (
       aRes->mGrid.all_pts(),
       mGrid.in_proj() * (ModeMapping ? aChScale : 1.0 ),
       aRes->mGrid.out()
   );

   return aRes;
}

const Pt2dr &  RImGrid::Step() const      {return mStepGr;}
Pt2dr RImGrid::Origine() const   {return mP0;}

Im2D_REAL8 RImGrid::DataGrid() {return mGrid;}


/**************************************************/
/*                                                */
/*                BufferedImGr                    */
/*                                                */
/**************************************************/

class BufferedImGr
{
      public :
         BufferedImGr
         (
             Pt2di aP0,
             Pt2di aP1,
             INT   aStepGr,
             Fonc_Num aFonc
         );
         ~BufferedImGr();

        Pt2di P0() const;
        Pt2di P1() const;

       inline void NewLine(Pt2di);
       inline void IncrX();
       REAL  GetVal() const {return mCurVal;}
       Pt2di PCur() const {return mPcur;}

      private :

          inline void ActualiseVals();
          REAL                 mCurVal;
          REAL                 mCurDeltaX;
          Pt2di                mPcur;


          
          
          Pt2di                mP0;
          Pt2di                mP1;
          INT                  mStepGr;
          INT                  mSqStep;
          Pt2di                mSzIm;
          Im2D_REAL8           mGrid;
          REAL **              mData;
};

BufferedImGr::~BufferedImGr() {}

void BufferedImGr::ActualiseVals()
{
    
    INT xo   =  mPcur.x /  mStepGr;
    INT yo   =  mPcur.y /  mStepGr;


    INT  px1 = mPcur.x - (xo*mStepGr);
    INT  py1 = mPcur.y - (yo*mStepGr);
    INT  px0 = mStepGr  -px1;
    INT  py0 = mStepGr  -py1;

    REAL * l0 = mData[yo]+xo;
    REAL * l1 = mData[yo+1]+xo;

    mCurVal = (
                    (px0 * py0) * l0[0]
                  + (px1 * py0) * l0[1]
                  + (px0 * py1) * l1[0]
                  + (px1 * py1) * l1[1]
               ) / mSqStep;

     mCurDeltaX =  (py0 * (l0[1]-l0[0]) +  py1 * (l1[1]-l1[0])) / mSqStep;


}

void BufferedImGr::NewLine(Pt2di aP)
{
     mPcur = aP - mP0;
     ActualiseVals();
}

void BufferedImGr::IncrX()
{
    mPcur.x++;
    if (mPcur.x % mStepGr==0) 
       ActualiseVals();
     else
       mCurVal += mCurDeltaX;
}

Pt2di BufferedImGr::P0() const {return mP0;}
Pt2di BufferedImGr::P1() const {return mP1;}



BufferedImGr::BufferedImGr
(
       Pt2di aP0,
       Pt2di aP1,
       INT   aStepGr,
       Fonc_Num aFonc
)  :
   mP0     (aP0),
   mP1     (aP1),
   mStepGr (aStepGr),
   mSqStep (ElSquare(mStepGr)),
   mSzIm   ((aP1-aP0) /aStepGr + Pt2di(3,3)),
   mGrid   (mSzIm.x,mSzIm.y), 
   mData   (mGrid.data())
{
     ELISE_COPY
     (
         mGrid.all_pts(),
         aFonc[Virgule(FX*mStepGr+aP0.x,FY*mStepGr+aP0.y)],
         mGrid.out()
     );
}


#define NBB_RGRC 8
template <class Type,class TyBase>  
         class  RLEImGridReechComp : public Fonc_Num_Comp_TPL<REAL>

{
     public :

             RLEImGridReechComp
             (
                  const Arg_Fonc_Num_Comp & anArg,
                  Im2D<Type,TyBase>  anIm,
                  BufferedImGr             aGrX,
                  BufferedImGr             aGrY,
                  REAL               aDef,
                  bool               aWithDef
             ) :
               Fonc_Num_Comp_TPL<REAL>(anArg,1,anArg.flux()),
               mIm         (anIm),
               mTIm        (anIm),
               mGrX        (aGrX),
               mGrY        (aGrY),
               mDef        (aDef),
               mWithDef    (aWithDef)
             {
             }
           
     private :
             typedef ElPFixed<NBB_RGRC> tPt;

             const class Pack_Of_Pts * values(const class Pack_Of_Pts *);

             Im2D<Type,TyBase>  mIm;
             TIm2D<Type,TyBase> mTIm;
             BufferedImGr             mGrX;
             BufferedImGr             mGrY;
             REAL               mDef;
             bool               mWithDef;
};


template <class Type,class TyBase>  
         const Pack_Of_Pts *  RLEImGridReechComp<Type,TyBase>::values
                              (
                                   const class Pack_Of_Pts * aGenPack
                               )
{
     
      const RLE_Pack_Of_Pts *  aRle = aGenPack->rle_cast ();
      INT aNb = aRle->nb();
      _pack_out->set_nb(aNb);


      if (aNb)
      {
         REAL * coord = _pack_out->_pts[0];
         Pt2di aP0 (aRle->vx0(),aRle->y());

         mGrX.NewLine(aP0);
         mGrY.NewLine(aP0);

         for (INT k=0 ; k<aNb ; k++)
         {
               Pt2dr  aP(mGrX.GetVal(),mGrY.GetVal());
               tPt aPFix(aP);

 
               coord[k] = TImGet<Type,TyBase,NBB_RGRC>::getr(mTIm,aPFix,mDef);
               mGrX.IncrX();
               mGrY.IncrX();
         }
      }

      return _pack_out;

}




template <class Type,class TyBase>  
         class  ImGridReechNotComp : public Fonc_Num_Not_Comp

{
     public :
           ImGridReechNotComp
           (
               Im2D<Type,TyBase>  anIm,
               BufferedImGr             aGrX,
               BufferedImGr             aGrY,
               REAL               aDef,
               bool               aWithDef
           )   :
               mIm      (anIm),
               mGrX     (aGrX),
               mGrY     (aGrY),
               mDef     (aDef),
               mWithDef (aWithDef)
           {
           }

     private :

         virtual  Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp & anArg)
         {
              ELISE_ASSERT(anArg.flux()->type() ==  Pack_Of_Pts::rle,"Not RLE in Im Grid");
              ELISE_ASSERT(anArg.flux()->dim() ==  2,"Dim !=2 in Grid");


              Fonc_Num_Computed * aRes =   new RLEImGridReechComp<Type,TyBase>
                                              (
                                                 anArg,
                                                 mIm,
                                                 mGrX, mGrY,
                                                 mDef,mWithDef
                                               );

              INT t0[2],t1[2];
              mGrX.P0().to_tab(t0);
              mGrX.P1().to_tab(t1);


              return clip_fonc_num_def_val
                     (
                           anArg,aRes,anArg.flux(),
                           t0,t1, mDef
                     );

         }

         virtual bool integral_fonc(bool /*integral_flux*/) const
         {
                 return false;
         }
         virtual INT dimf_out() const
         {
                 return 1;
         }

         void VarDerNN(ElGrowingSetInd &) const{ELISE_ASSERT(false,"No VarDerNN")};

         virtual Fonc_Num deriv(INT k) const
         {
                 ELISE_ASSERT(false,"No formal derivation for GridIm");
                 return 0;
         }
         virtual void  show(ostream & os) const
         {
                 os << "[GridIm]";
         }
         REAL ValFonc(const PtsKD &) const
         {
                 ELISE_ASSERT(false,"No ValFonc for GridIm");
                 return 0;
         }


         Im2D<Type,TyBase>  mIm;
         BufferedImGr             mGrX;
         BufferedImGr             mGrY;
         REAL               mDef;
         bool               mWithDef;
};


template <class Type,class TyBase>  
         Fonc_Num Im2D<Type,TyBase>::ImGridReech
                  (
                        Fonc_Num reechantX,
                        Fonc_Num reechantY,
                        INT      aStepGr,
                        Pt2di    aP0,
                        Pt2di    aP1,
                        REAL     def
                  )
{
    BufferedImGr aGrX(aP0,aP1,aStepGr,reechantX);
    BufferedImGr aGrY(aP0,aP1,aStepGr,reechantY);

    return new ImGridReechNotComp<Type,TyBase>
               (
                   *this,aGrX,aGrY,
                   def, true
                );
}

template <class Type,class TyBase>  
         Fonc_Num Im2D<Type,TyBase>::ImGridReech
                  (
                        Fonc_Num reechantX,
                        Fonc_Num reechantY,
                        INT      sz_grid,
                        REAL     def
                   )
{
    return ImGridReech(reechantX,reechantY,sz_grid,Pt2di(0,0),sz(),def);
}

template class Fonc_Num Im2D<U_INT1,INT>::ImGridReech
                  (
                        Fonc_Num reechantX,
                        Fonc_Num reechantY,
                        INT      sz_grid,
                        REAL     def
                   );
/*
Fonc_Num Instantiate_ImGridReech()
{
   Im2D<U_INT1,INT> anIm(1,1);
   return anIm.ImGridReech(FX,FY,1,2.0);
}
*/

/**************************************************/
/*                                                */
/*               PtImGrid                         */
/*                                                */
/**************************************************/


PtImGrid::PtImGrid
(
     bool AdaptStep,
     Pt2dr aP0, 
     Pt2dr aP1, 
     Pt2dr  aStepGr,
     const std::string & aName
) :
    mGX (new RImGrid(AdaptStep,aP0,aP1,aStepGr,aName+"_X")),
    mGY (new RImGrid(AdaptStep,aP0,aP1,aStepGr,aName+"_Y")),
    mName (aName)
{
}



PtImGrid::PtImGrid(RImGrid * aGX,RImGrid * aGY,const std::string & aName) :
    mGX   (aGX),
    mGY   (aGY),
    mName (aName)
{
}

Pt2dr PtImGrid::ToGrid(Pt2dr aP) const {return mGY->ToGrid(aP);}
Pt2dr PtImGrid::ToReal(Pt2dr aP) const {return mGX->ToReal(aP);}
Pt2di PtImGrid::SzGrid()         const {return mGX->SzGrid();}
bool  PtImGrid::StepAdapted()    const {return mGX->StepAdapted();}


const std::string & PtImGrid::Name() const {return mName;}
const std::string & PtImGrid::NameX()const  {return mGX->Name();}
const std::string & PtImGrid::NameY()const  {return mGY->Name();}

Pt2dr  PtImGrid::Value(Pt2dr aP)
{
    return Pt2dr(mGX->Value(aP),mGY->Value(aP));
}


Pt2dr  PtImGrid::ValueAndDer(Pt2dr aRealP,Pt2dr & aGradX,Pt2dr & aGradY)
{
  Pt2dr aRes;
  aRes.x = mGX->ValueAndDer(aRealP,aGradX);
  aRes.y = mGY->ValueAndDer(aRealP,aGradY);


  return aRes;
}


void  PtImGrid::SetValueGrid(Pt2di anInd,Pt2dr  aV)
{
    mGX->SetValueGrid(anInd,aV.x);
    mGY->SetValueGrid(anInd,aV.y);
}

void PtImGrid::write(ELISE_fp & aFile) const
{
     mGX->write(aFile);
     mGY->write(aFile);
}

PtImGrid::PtImGrid(ELISE_fp & aFile) :
   mGX (RImGrid::read(aFile)),
   mGY (RImGrid::read(aFile))
{
}

PtImGrid * PtImGrid::read(ELISE_fp & aFile)
{
    return new PtImGrid(aFile);
}

PtImGrid::~PtImGrid()
{
   delete mGX;
   delete mGY;
}

void  PtImGrid::SetTrChScaleOut(REAL aChScale,Pt2dr aTr)
{
  mGX->SetTrChScaleOut(aChScale,aTr.x);
  mGY->SetTrChScaleOut(aChScale,aTr.y);
}

void  PtImGrid::SetTrChScaleIn(REAL aChScale,Pt2dr aTr)
{
  mGX->SetTrChScaleIn(aChScale,aTr);
  mGY->SetTrChScaleIn(aChScale,aTr);
}


const Pt2dr &  PtImGrid::P0() const {return mGX->P0();}
const Pt2dr &  PtImGrid::P1() const {return mGX->P1();}



const Pt2dr &  PtImGrid::Step() const      {return mGX->Step();}
Pt2dr PtImGrid::Origine() const   {return mGX->Origine();}

Im2D_REAL8 PtImGrid::DataGridX() {return mGX->DataGrid();}
Im2D_REAL8 PtImGrid::DataGridY() {return mGY->DataGrid();}

/**************************************************/
/*                                                */
/*               cDbleGrid                        */
/*                                                */
/**************************************************/

cDbleGrid::cXMLMode::cXMLMode(bool ToSwap) :
   toSwapDirInv (ToSwap)
{
}


bool BugDG = false;

cDbleGrid::cDbleGrid(PtImGrid *aDir,PtImGrid * anInv) :
   pGrDir (aDir),
   pGrInv (anInv)
{
}

bool   cDbleGrid::StepAdapted()  const {return pGrDir->StepAdapted();}

Pt2dr cDbleGrid::PP()
{
     return Inverse(Pt2dr(0,0));
}

REAL cDbleGrid::Focale()
{
    Pt2dr aPP = PP();
    Pt2dr aP1 = Direct(aPP+Pt2dr(1,0));
    Pt2dr aP2 = Direct(aPP+Pt2dr(-1,0));
    return  2.0 / euclid(aP1,aP2);
}


PtImGrid * FromXMLExp(const cGridDeform2D & aGr)
{
    return new PtImGrid
           (
	      new RImGrid(aGr.Origine(),aGr.Step(),aGr.ImX()),
	      new RImGrid(aGr.Origine(),aGr.Step(),aGr.ImY()),
	      "toto"
	   );
}

cDbleGrid::cDbleGrid
(
      const cGridDirecteEtInverse & aGDEI
) :
  pGrDir (FromXMLExp(aGDEI.Directe())),
  pGrInv (FromXMLExp(aGDEI.Inverse()))
{
}

cDbleGrid::cDbleGrid
(
    cXMLMode aXM,
    const std::string & aDir,
    const std::string & aXML
)
{
   cElXMLTree aTree(aDir+aXML);
   cElXMLTree * aTrNew = aTree.Get("GridDirecteEtInverse");
   if (aTrNew)
   {
      cGridDirecteEtInverse aGDEI;
      xml_init(aGDEI,aTrNew);

      pGrDir = FromXMLExp(aGDEI.Directe());
      pGrInv = FromXMLExp(aGDEI.Inverse());

   }
   else
   {
       pGrInv = new PtImGrid (aTree.GetUnique("grid_inverse")->GetPtImGrid(aDir));
       pGrDir = new PtImGrid (aTree.GetUnique("grid_directe")->GetPtImGrid(aDir));
       if (aXM.toSwapDirInv)
       {
          ElSwap(pGrInv,pGrDir);
       }
   }
}
 
cDbleGrid::cDbleGrid
(
    bool P0P1IsBoxDirect,
    bool  AdaptStep,
    Pt2dr aP0In,Pt2dr aP1In,
    Pt2dr               aStepDir,
    ElDistortion22_Gen & aDist,
    const std::string &  aName,
    bool                 doDir,
    bool                 doInv
)   :
    mName (aName)
{

// NIKRUP
// std::cout << "cDbleGrid::cDbleGrid " << aP0In << aP1In << aStepDir << "\n";

    PtImGrid * aGR1 = new PtImGrid(AdaptStep,aP0In,aP1In,aStepDir,mName+ (P0P1IsBoxDirect ? "_Directe" : "_Inverse"));
// std::cout << "DONE PtImGridPtImGrid\n";
    Pt2di aSzGr = aGR1->SzGrid();

    bool First = true;
    Pt2dr aP0Dist(1e9,1e9);
    Pt2dr aP1Dist(-1e9,-1e9);

    for (INT aI=0; aI<aSzGr.x ; aI++)
    {
        for (INT aJ=0; aJ<aSzGr.y ; aJ++)
        {
            Pt2di aPIJ(aI,aJ);
            Pt2dr aPR = aGR1->ToReal(Pt2dr(aPIJ));
            Pt2dr  aPDR = P0P1IsBoxDirect ?  aDist.Direct(aPR) : aDist.Inverse(aPR) ;
            aGR1->SetValueGrid(aPIJ,aPDR);
            if (First)
            {
                aP0Dist = aPDR;
                aP1Dist = aPDR;
                First = false;
            }
            else
            {
                aP0Dist.SetInf(aPDR);
                aP1Dist.SetSup(aPDR);
            }
        }
    }

    
    PtImGrid * aGR2 = 0;
    if (doInv)
    {

       Pt2dr aStepInv = (aP1Dist-aP0Dist).dcbyc(Pt2dr(aSzGr));
    /*
    REAL aStepInv = sqrt
                    (
                         ((aP1Dist.x-aP0Dist.x)*(aP1Dist.y-aP0Dist.y) )
                       / (aSzGr.x*aSzGr.y)
                    );
    */
       aGR2 = new PtImGrid(AdaptStep,aP0Dist,aP1Dist,aStepInv,aName+ (P0P1IsBoxDirect ? "_Inverse" : "_Directe"));
       aSzGr = aGR2->SzGrid();

       for (INT aI=0; aI<aSzGr.x ; aI++)
       {
           for (INT aJ=0; aJ<aSzGr.y ; aJ++)
	   {
               Pt2di aPIJ(aI,aJ);
               Pt2dr aPR = aGR2->ToReal(Pt2dr(aPIJ));
               Pt2dr  aPDR = P0P1IsBoxDirect?  aDist.Inverse(aPR) : aDist.Direct(aPR) ;
               aGR2->SetValueGrid(aPIJ,aPDR);
	   }
       }
    }

    pGrDir  =  P0P1IsBoxDirect ? aGR1 : aGR2;
    pGrInv  =  P0P1IsBoxDirect ? aGR2 : aGR1;

    if (!doDir)
    {
        delete pGrDir;
        pGrDir = 0;
    }
}

cDbleGrid::~cDbleGrid()
{
   delete pGrDir;
   delete pGrInv;
}

Pt2dr cDbleGrid::Direct(Pt2dr aP) const  
{
   ELISE_ASSERT(pGrDir,"cDbleGrid::Direct");
   return pGrDir->Value(aP);
}
bool cDbleGrid::OwnInverse(Pt2dr & aP) const     
{
   ELISE_ASSERT(pGrInv,"cDbleGrid::OwnInverse");
   aP = pGrInv->Value(aP);
   return true;
}

Pt2dr   cDbleGrid::ValueAndDer(Pt2dr aRealP,Pt2dr & aGradX,Pt2dr & aGradY)
{
   
   ELISE_ASSERT(pGrDir,"cDbleGrid::Direct");
   return pGrDir->ValueAndDer(aRealP,aGradX,aGradY);
}

void  cDbleGrid::Diff(ElMatrix<REAL> &,Pt2dr) const
{
   ELISE_ASSERT(false,"cDbleGrid::Diff");
}

void cDbleGrid::write(ELISE_fp & aFile)
{
    pGrDir->write(aFile);
    pGrInv->write(aFile);
}

void cDbleGrid::write(const  std::string & aName)
{
  ELISE_fp aFile(aName.c_str(),ELISE_fp::WRITE);
  write(aFile);
  aFile.close();
}

const Pt2dr &  cDbleGrid::P0_Dir()   const {return pGrDir->P0();}
const Pt2dr &  cDbleGrid::P1_Dir()   const {return pGrDir->P1();}
const Pt2dr &  cDbleGrid::Step_Dir() const {return pGrDir->Step();}


const  PtImGrid & cDbleGrid::GrDir() const  {return *pGrDir;}
const  PtImGrid & cDbleGrid::GrInv() const  {return *pGrInv;}
PtImGrid & cDbleGrid::GrDir() {return *pGrDir;}
PtImGrid & cDbleGrid::GrInv() {return *pGrInv;}

cDbleGrid * cDbleGrid::read(ELISE_fp & aFile)
{
  PtImGrid * pDir = PtImGrid::read(aFile);
  PtImGrid * pInv = PtImGrid::read(aFile);
  return new cDbleGrid(pDir,pInv);
}
cDbleGrid * cDbleGrid::read(const  std::string & aName)
{
     ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);
     cDbleGrid * aRes = read(aFile);
     aFile.close();

     return aRes;
}


void  cDbleGrid::SetTrChScaleDir(REAL aChScale,Pt2dr aTr)
{
    pGrDir->SetTrChScaleOut(aChScale,aTr);
    pGrInv->SetTrChScaleIn(aChScale,aTr);
}

void  cDbleGrid::SetTrChScaleInv(REAL aChScale,Pt2dr aTr)
{
    pGrInv->SetTrChScaleOut(aChScale,aTr);
    pGrDir->SetTrChScaleIn(aChScale,aTr);
}



void cDbleGrid::SauvDataGrid
     (
         const std::string &  aNameDir,
         Im2D_REAL8 anIm,
         const std::string & aName
     )
{
     std::string aFullName = aNameDir + aName + ".dat";
     ELISE_fp  aFp(aFullName.c_str(),ELISE_fp::WRITE);
     aFp.write(anIm.data_lin(),sizeof(double),anIm.tx()*anIm.ty());
     aFp.close();
}


void cDbleGrid::PutXMWithData
     (
         cElXMLFileIn &       aFileXML,
         const std::string &  aNameDir
     )
{
    aFileXML.PutDbleGrid(false,*this);
    SauvDataGrid(aNameDir,GrDir().DataGridX(),GrDir().NameX());
    SauvDataGrid(aNameDir,GrDir().DataGridY(),GrDir().NameY());
    SauvDataGrid(aNameDir,GrInv().DataGridX(),GrInv().NameX());
    SauvDataGrid(aNameDir,GrInv().DataGridY(),GrInv().NameY());
}


void cDbleGrid::SaveXML(const std::string & aName)
{
    cGridDirecteEtInverse aXMLGr = ToXMLExp(*this);
    MakeFileXML<cGridDirecteEtInverse>(aXMLGr,aName);

}


cDbleGrid *  cDbleGrid::StdGridPhotogram(const std::string & aNameFile,int aSzDisc)
{
   cElXMLTree aTree(aNameFile);

   if (aTree.Get("doublegrid"))
   {
      std::string aDir,aFile;
      SplitDirAndFile(aDir,aFile,aNameFile);

      cDbleGrid::cXMLMode aXmlMode;
      return new  cDbleGrid(aXmlMode,aDir,aFile);
   }

   if (aTree.Get("CalibrationInternConique"))
   {
        ElCamera * aCam = Cam_Gen_From_File(aNameFile,"CalibrationInternConique",(cInterfChantierNameManipulateur*)0);
        cDistStdFromCam aD(*aCam);

        cDbleGrid * aRes = new cDbleGrid
                              (
                                  false, // P0P1 Direct non par defaut M->C
			          true,
				  Pt2dr(0,0),Pt2dr(aCam->Sz()),
				  Pt2dr(aSzDisc,aSzDisc),
				  aD
			      );
        delete aCam;

        return aRes;
   }



   ELISE_ASSERT(false,"cDbleGrid::StdGridPhotogram");
   return 0;
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
