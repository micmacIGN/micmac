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


std::vector<double> MakeVec1(const double & aVal)
{
    std::vector<double>  aRes;
    aRes.push_back(aVal);
    return aRes;
}


// dlopen etc ....

int cSsBloc::theCptGlob = 0;

cSsBloc::cSsBloc (int aI0,int aI1) :
   mI0  (aI0),
   mNb  (aI1-aI0),
   mInt (0),
   mCpt (theCptGlob++)
{
}

void cSsBloc::BlocSetInt(cIncIntervale & anInt) 
{
   mInt = &anInt;
}

/*
*/

void cSsBloc::AssertIntInit() const
{
   if (mInt==0)
   {
       std::cout << "For Cpt = " << mCpt << "\n";
       ELISE_ASSERT(false,"cSsBloc::AssertIntInit");
   }
}

int  cSsBloc::I0Brut() const
{
   return mI0;
}
int  cSsBloc::I1Brut() const
{
   return mI0+mNb;
}
int  cSsBloc::Nb() const
{
   return mNb;
}

int cSsBloc::I0AbsAlloc() const
{
    AssertIntInit();
    return mI0 + mInt->I0Alloc();
}

int cSsBloc::I1AbsAlloc() const
{
    AssertIntInit();
    return I0AbsAlloc()+mNb;
}


int cSsBloc::I0AbsSolve() const
{
    AssertIntInit();
    return mI0 + mInt->I0Solve();
}
int cSsBloc::I1AbsSolve() const
{
    AssertIntInit();
    return I0AbsSolve()+mNb;
}



bool cSsBloc::operator ==(const cSsBloc & aSB) const
{
   return (I0AbsAlloc() == aSB.I0AbsAlloc()) &&  (I1AbsAlloc() == aSB.I1AbsAlloc());
}

bool cSsBloc::disjoint (const cSsBloc & aSB) const
{
   return (I0AbsAlloc() >= aSB.I1AbsAlloc()) || (I1AbsAlloc() <= aSB.I0AbsAlloc());
}


int cSsBloc::Cpt() const
{
   return mCpt;
}


/*********************************************************/
/*                                                       */
/*               cIncIntervale                           */
/*                                                       */
/*********************************************************/

cIncIntervale::cIncIntervale
(
    bool                   isTmp,
    const tId &            anId,
    cSetEqFormelles &       aSet,
    int                    aNbInc
) :
      mI0Alloc (aSet.Alloc().CurInc()),
      mI1Alloc ((aNbInc<=0) ? -1: (mI0Alloc+aNbInc)),
      mI0Solve (-1),
      mI1Solve (-1),
      mSet     (&aSet),
      mId      (anId),
      mNumBlocAlloc (-1),
      mNumBlocSolve  (-1),
      mFiged        (false),
      mOrder        (-1),
      mIsTmp        (isTmp)
{
}


cSsBloc cIncIntervale::SsBlocComplet()
{
    cSsBloc aRes(0,Sz());
    aRes.BlocSetInt(*this);
    return aRes;
}

bool cIncIntervale::IsTmp() const
{
   return mIsTmp;
}

double cIncIntervale::Order() const
{
   return mOrder;
}
void cIncIntervale::SetOrder(double anOrder) 
{
   mOrder = anOrder;
}
void cIncIntervale::SetFirstIntervalSolve()
{
   mNumBlocSolve = 0;
   mI0Solve = 0;
   mI1Solve = mI1Alloc-mI0Alloc;
}

void cIncIntervale::InitntervalSolve(const cIncIntervale & anI)
{
   mNumBlocSolve = anI.mNumBlocSolve+1;
   mI0Solve = anI.mI1Solve;
   mI1Solve = mI0Solve +  mI1Alloc-mI0Alloc;
}


/*
int cIncIntervale::NumBloc() const
{
   return mNumBloc;
}
*/


void cIncIntervale::SetNumAlloc(int aNum)
{
   ELISE_ASSERT((mNumBlocAlloc==-1)&& (aNum>=0),"cIncIntervale::SetNum");
   mNumBlocAlloc = aNum;
}

bool cIncIntervale::operator == (const cIncIntervale & anII) const
{
    return (mI0Alloc==anII.mI0Alloc) && (mI1Alloc==anII.mI1Alloc);
}


cIncIntervale::cIncIntervale(const tId & anId,INT anI0,INT anI1,bool isTmp) :
      mI0Alloc (anI0),
      mI1Alloc (anI1),
      mI0Solve (-1),
      mI1Solve (-1),
      mSet   (0),
      mId    (anId),
      mNumBlocAlloc (-1),
      mNumBlocSolve  (-1),
      mFiged (false),
      mOrder        (-1),
      mIsTmp        (isTmp)
{
}

cIncIntervale::cIncIntervale(const cIncIntervale & anInterv,const tId & anId) :
      mI0Alloc    (anInterv.mI0Alloc),
      mI1Alloc    (anInterv.mI1Alloc),
      mI0Solve    (anInterv.mI0Solve),
      mI1Solve    (anInterv.mI1Solve),
      mSet        (anInterv.mSet  ),
      mId         (anId),
      mNumBlocAlloc (anInterv.mNumBlocAlloc),
      mNumBlocSolve  (anInterv.mNumBlocSolve),
      mFiged (false),
      mOrder        (-1)
{
}

void cIncIntervale::SetFiged(bool aFg)
{
   mFiged = aFg;
}

bool cIncIntervale::IsFiged() const
{
   return mFiged;
}

void cIncIntervale::SetName(const tId & anId)
{
  mId = anId;
}

void cIncIntervale::Close()
{
   ELISE_ASSERT(mI1Alloc==-1,"Multiple Clode in cIncIntervale");
   mI1Alloc = mSet->Alloc().CurInc();
}

INT cIncIntervale::I0Alloc() const
{
   ELISE_ASSERT(mI1Alloc!=-1,"Use of UnCloded in cIncIntervale");
   return mI0Alloc;
}
INT cIncIntervale::I1Alloc() const
{
   ELISE_ASSERT(mI1Alloc!=-1,"Use of UnCloded in cIncIntervale");
   return mI1Alloc;
}

INT cIncIntervale::I0Solve() const
{
   ELISE_ASSERT(mI0Solve!=-1,"Use of UnCloded in cIncIntervale");
   return mI0Solve;
}
INT cIncIntervale::I1Solve() const
{
   ELISE_ASSERT(mI1Solve!=-1,"Use of UnCloded in cIncIntervale");
   return mI1Solve;
}



INT cIncIntervale::Sz() const
{
   ELISE_ASSERT(mI1Alloc!=-1,"Use of UnCloded in cIncIntervale");
   return mI1Alloc-mI0Alloc;
}

const cIncIntervale::tId & cIncIntervale::Id() const
{
    return mId;
}

bool cIncIntervale::Overlap(const cIncIntervale & anInt) const
{
    return   (mI0Alloc <  anInt.mI1Alloc) && (anInt.mI0Alloc <  mI1Alloc);
}

void cIncIntervale::SetI0I1Alloc(INT I0,INT I1)
{
    mI0Alloc = I0;
    mI1Alloc = I1;
}


/*********************************************************/
/*                                                       */
/*               cIncIntervale                           */
/*                                                       */
/*********************************************************/

/*
class cIdIntCmp
{
     public :
         bool operator()(const cIncIntervale & anII1,const cIncIntervale & anII2) const
         {
              return anII1.Id() < anII2.Id();
         }
};

class cMapIncInterv  : public std::set<cIncIntervale,cIdIntCmp>
{
};
*/
bool cIdIntCmp::operator()(const cIncIntervale & anII1,const cIncIntervale & anII2) const
{
    return anII1.Id() < anII2.Id();
}

cIncListInterv::~cIncListInterv() 
{
   // delete mMap;
}


bool cIncListInterv::Equal(const cIncListInterv& anILI) const
{
   return mMap==anILI.mMap;
   // return (std::set<cIncIntervale,cIdIntCmp>)*mMap==(std::set<cIncIntervale,cIdIntCmp>)*anILI.mMap;
}

typedef std::set<cIncIntervale,cIdIntCmp>::const_iterator  tCSetIII;

void cIncListInterv::Init()
{
   mI0Min    =1000000000;
   mI1Max    =-1;
   mSurf     =0;
   mMap      = cMapIncInterv();
   mMayOverlap    = false;
}
bool cIncListInterv::MayOverlap () const {return mMayOverlap;}
cIncListInterv::cIncListInterv() 
{ 
   Init();
}

cIncListInterv::cIncListInterv(bool isTmp,const cIncIntervale::tId & anId,INT anI0,INT anI1)
{
   Init();
   cIncIntervale anInt(anId,anI0,anI1,isTmp);
   AddInterv(anInt);
}


void cIncListInterv::AddInterv(const cIncIntervale & anInterv,bool CanOverlap)
{
   if (anInterv.Sz()==0 )
      return;
   for (tCSetIII anIt =  mMap.begin() ; anIt!= mMap.end() ; anIt++)
   {
       if (anIt->Id() == anInterv.Id())
       {
           std::cout << "For Id=" << anIt->Id() << "\n";
           ELISE_ASSERT
           (
               anIt->Id() != anInterv.Id(),
               "Ambiguous Key in cIncListInterv::AddInterv"
           );
       }
       if (CanOverlap)
          mMayOverlap = true;
       else
       {
           ELISE_ASSERT
           (
                ! (anInterv.Overlap(*anIt)),
               "Overlapping interval in cIncListInterv::AddInterv"
           );
       }
   }

   mMap.insert(anInterv);
   ElSetMin(mI0Min,anInterv.I0Alloc());
   ElSetMax(mI1Max,anInterv.I1Alloc());
   mSurf += anInterv.Sz();
}

bool cIncListInterv::IsConnexe0N() const
{
    return (mI0Min==0) && (mI1Max==mSurf);
}


const cIncIntervale & cIncListInterv::FindEquiv(const cIncIntervale & anInterv) const
{
   tCSetIII anIt = mMap.find(anInterv);
   ELISE_ASSERT
   (
       anIt!=mMap.end(),
       "Cant find required key in cIncListInterv::FindEquiv"
   );
   ELISE_ASSERT
   (
       anIt->Sz() ==anInterv.Sz(),
       "Incompatible interval in cIncListInterv::FindEquiv"
   );

   return *anIt;
}

void cIncListInterv::ResetInterv(const cIncIntervale & anInterv)
{
   mMayOverlap = true;
   ELISE_ASSERT
   (
        anInterv.I1Alloc()<=mI1Max,
        "Increase I1max in cIncListInterv::ResetInterv"
    );

   cIncIntervale & It = const_cast<cIncIntervale &>
	                (cIncListInterv::FindEquiv(anInterv));
   It.SetI0I1Alloc(anInterv.I0Alloc(),anInterv.I1Alloc());
}

const cMapIncInterv & cIncListInterv::Map() const
{
    return mMap;
}


INT cIncListInterv::I1Max()  const {return mI1Max;}
INT cIncListInterv::I0Min()  const {return mI0Min;}
INT cIncListInterv::Surf() const {return mSurf;}

/*********************************************************/
/*                                                       */
/*            cECFN_SetString                            */
/*                                                       */
/*********************************************************/

class cECFNSS_Cmp
{
     public :
         bool operator()(const cVarSpec  & anII1,const cVarSpec  & anII2) const
         {
              return anII1.Name() < anII2.Name();
         }
};


class cECFN_SetString  : public std::set<cVarSpec,cECFNSS_Cmp>
{
    public :
    private :
};


/*********************************************************/
/*                                                       */
/*             cDico_SymbFN                              */
/*                                                       */
/*********************************************************/

class cCelDico_SFN
{
    public :
	    cCelDico_SFN () :
		    mSymbPut (false),
		    mNbRef (0)
	    {
	    }

	    bool  mSymbPut;
	    INT   mNum;
	    INT   mNbRef;
	    std::string NameSymb() const 
	    {
                return "tmp"+ToString(mNum) +"_";
	    }
};

class cDico_SymbFN_CMP
{
     public :
         bool operator()
              (const Fonc_Num  & aF1,const Fonc_Num  & aF2) const
         {
              return aF1.CmpFormel(aF2) == -1;
         }
};



class  cDico_SymbFN 
{
	public :
	    cDico_SymbFN();
            void AddF(Fonc_Num aF);	
            bool SqueezComp(Fonc_Num aF);
            void PutFonc(Fonc_Num  aF,cElCompileFN &);
	    void PutSymbs(cElCompileFN &);
	private :


         typedef std::map<Fonc_Num,cCelDico_SFN,cDico_SymbFN_CMP> tDic;
	 typedef tDic::value_type  tPair;
	 typedef tDic::iterator    tIter;


         tDic  mDic;
	 INT   mNumSymb;
	 std::vector<tPair *>  mVSymb;


};


void cDico_SymbFN::PutSymbs(cElCompileFN & anEnv)
{
   for (INT aK=0 ; aK<INT(mVSymb.size()) ; aK++)
   {
       tPair * aPair = mVSymb[aK];
       Fonc_Num aPF = aPair->first;
       anEnv << "   double "
	     << aPair->second.NameSymb() 
	     << " = "   
	     << aPF << ";" ;
       if (aPF.HasNameCpp())
          anEnv << " // " << aPF.NameCpp();
       anEnv << "\n";
       aPair->second.mSymbPut = true;
   }
   anEnv << "\n";
}


cDico_SymbFN::cDico_SymbFN() :
    mNumSymb (0)
{
}

void  cDico_SymbFN::AddF(Fonc_Num aF)
{
   Fonc_Num::tKindOfExpr aK = aF.KindOfExpr();

   if (
             (aK == Fonc_Num::eIsICste)
          || (aK == Fonc_Num::eIsRCste)
          || (aK == Fonc_Num::eIsVarSpec)
      )
      return;
   
   cCelDico_SFN & aCel = mDic[aF];

   // Lorsque l'expression est marquee comme interessante,
   // on force l'emission de symbole pour clarifier
   if (aF.HasNameCpp())
      aCel.mNbRef++;
   aCel.mNbRef++;
   if ((!aF.IsVarSpec()) && (aCel.mNbRef == 2))
   {
      aCel.mNum = mNumSymb++;
      tPair * aPair = &(*mDic.find(aF));
      mVSymb.push_back(aPair);
   }
}

bool cDico_SymbFN::SqueezComp(Fonc_Num aF)
{
   tIter anIt = mDic.find(aF);
   return (anIt!=mDic.end());
}

/*********************************************************/
/*                                                       */
/*            cElCompiledFonc                            */
/*                                                       */
/*********************************************************/

template <class Type> void SetSizeVect(std::vector<Type> & aVect,INT aSz,const Type & aDef= Type())
{
    aVect.clear();
    for (INT aK=0; aK<aSz ; aK++)
        aVect.push_back(aDef);
}

cElCompiledFonc::cElCompiledFonc(INT aDimOut)  :
    mDimOut          (aDimOut),
    isValComputed    (false),
    isDerComputed    (false),
    isCoordSet       (false),
    isCurMappingSet  (false),
    mAlwaysIndexed   (false),
    mCompDer         (aDimOut),
    mVal             (aDimOut)
{
}

std::string & cElCompiledFonc::NameAlloc()
{
   return mNameAlloc;
}


void cElCompiledFonc::AddIntRef(const cIncIntervale & anInterv)
{
   mMapRef.AddInterv(anInterv);
}

void cElCompiledFonc::CloseIndexed()
{
   mAlwaysIndexed = true;
   mNbCompVar = mMapRef.Surf();
   SetSizeVect(mCompCoord    ,mNbCompVar);
   for (INT aD = 0 ; aD<mDimOut ; aD++)
   {
      SetSizeVect(mCompDer[aD]      ,mNbCompVar);
   }
   SetNoInit();
}

void cElCompiledFonc::AsserNotAlwaysIndexed() const
{
   ELISE_ASSERT
   (
        ! mAlwaysIndexed,
        "cElCompiledFonc::AsserNotAlwaysIndexed"
   );
}

void cElCompiledFonc::SetNoInit()
{
   SetMappingCur(mMapRef,0);
   isCurMappingSet = false;
   isCoordSet      = false;
   isValComputed = false;
   isDerComputed = false;
}


void cElCompiledFonc::Close(bool Dyn)
{

   if (!Dyn)
   {
      if (!mMapRef.IsConnexe0N())
      {
          std::cout << "MAP REF; Min " << mMapRef.I0Min() 
                    << " ; Max " <<  mMapRef.I0Min() 
                    << " ; Surf " <<   mMapRef.Surf() << "\n";
          ELISE_ASSERT(false,"Bad Interv in cElCompiledFonc::Close");
      }
   }
   CloseIndexed();

}

cElCompiledFonc::~cElCompiledFonc() {}

int cElCompiledFonc::LIC(const int & i) const
{
    return i;
}


extern bool AllowUnsortedVarIn_SetMappingCur;

const cIncListInterv &  cElCompiledFonc::MapRef() const
{
   return mMapRef;
}



void cElCompiledFonc::SetMappingCur
     (
           const cIncListInterv & aList,
           cSetEqFormelles *      aSet
     )
{

      
     if (aList.Surf() != mMapRef.Surf())
     {
        std::cout << "LIST  :" << aList.Surf() << " ; REF : " <<  mMapRef.Surf() << "\n";
        ELISE_ASSERT
        (
          false,
          "Different Sur of Interv in cElCompiledFonc::SetMappingCur"
        );
     }

     //  On genere les indexe pontcuels
     {
        const cMapIncInterv & aSet = aList.Map();
        ELISE_ASSERT(mMapRef.IsConnexe0N(),"Bad Interv in SetMappingCur indexe");
        SetSizeVect(mMapComp2Real,mNbCompVar,-1);
        for (tCSetIII itCur = aSet.begin() ; itCur != aSet.end() ; itCur++)
        {
             const cIncIntervale & itRef = mMapRef.FindEquiv(*itCur);
             INT I0Comp = itRef.I0Alloc();
             INT sz = itRef.Sz();
             ELISE_ASSERT(itCur->Sz()==sz,"cElCompiledFonc::SetMappingCur Indexe");
             INT I0Real = itCur->I0Alloc();

             for (int aK=0 ; aK<sz ; aK++)
             {
                ELISE_ASSERT(mMapComp2Real.at(I0Comp+aK)==-1,"cElCompiledFonc::SetMappingCur Indexe");
                mMapComp2Real.at(I0Comp+aK) = I0Real+aK;
             }
        }
        // Le 12/05/2014 : j'ai l'impression que la tri de mMapComp2Real est une erreur (subsiste des anciens ??)
        // mais que jamais cree de pb car deja trie, le isSorted estfait pour verifier cela
        bool isSorted = true;
        for (int aK=0 ; aK<int(mMapComp2Real.size()) ; aK++)
        {
            ELISE_ASSERT(mMapComp2Real[aK]>=0,"cElCompiledFonc::SetMappingCur Indexe");
             if ((aK>=1) && (mMapComp2Real[aK] <= mMapComp2Real[aK-1]))
                 isSorted = false;
        }
        if (!isSorted)
        {
            if (! AllowUnsortedVarIn_SetMappingCur)
            {
                for (int aK=0 ; aK<20 ; aK++)
                    std::cout << "INTERNAL ERROR IN MICMAC : contact devlopment team\n";
                std::cout << "ERROR : bad assertion cElCompiledFonc::SetMappingCur\n";
                ElEXIT(-1,"INTERNAL ERROR IN MICMAC (cElCompiledFonc::SetMappingCur) : contact devlopment team");
            }
        }
        // A PRIORI CETTE LIGNE EST NEFASTE (en general INUTILE) .... a verifier a l'usage
        // std::sort(mMapComp2Real.begin(),mMapComp2Real.end());
     }
     // std::cout <<  "  ----SetMappingCur----   "  << aSet.size() << "\n";

     // On compile les intervales, pour la gestion en block
     {
         // 1 - on trie les intervalle de ref par ordre croissant d'indices
         std::vector<const cIncIntervale *> aVRef;
         const cMapIncInterv & aSetR = mMapRef.Map();
         for (tCSetIII itCur = aSetR.begin() ; itCur != aSetR.end() ; itCur++)
             aVRef.push_back(&(*itCur));

         cCmpI0PtrII aCmp; 
         std::sort(aVRef.begin(),aVRef.end(),aCmp);
         mBlocs.clear();
         for (int aK=0 ; aK<int(aVRef.size()) ; aK++)
         {
             const cIncIntervale & itCur = aList.FindEquiv(*aVRef[aK]);
             // NO::BlocSetInt
             mBlocs.push_back(cSsBloc(itCur.I0Alloc(),itCur.I1Alloc()));
         }
     }
     isCurMappingSet = true;

     if (aSet && aSet->IsClosed())
     {
       InitBloc(*aSet);
     }

// std::cout << "YURTEZIOPPP " <<
}

/*
class cCmpNumBlocSolve
{
    public : 
       bool operator () (const cSsBloc & aBl1,const cSsBloc & aBl2)
       {
            return aBl1.Int()->NumBlocSolve()  < aBl2.Int()->NumBlocSolve();
       }
};
*/

void   cElCompiledFonc::InitBloc(const cSetEqFormelles & aSet)
{
    int aSom=0; 
    for (int aKB=0 ; aKB<int(mBlocs.size()) ; aKB++)
    {
       mBlocs[aKB] = aSet.GetBlocOfI0Alloc(mBlocs[aKB].I0Brut(),mBlocs[aKB].I1Brut()); // ###
       // aSom += mBlocs[aKB].I1Abs() - mBlocs[aKB].I0Abs();
       aSom += mBlocs[aKB].Nb();
       //std::cout << "IIIIBloc " <<  (mBlocs[aKB].mInt->NumBloc()) << "\n";
   }
   // cCmpNumBlocSolve aCmp;
   // std::sort(mBlocs.begin(),mBlocs.end(),aCmp);
}


void cElCompiledFonc::SetCoordCur(const double * aRealCoord)
{

   ELISE_ASSERT(isCurMappingSet,"No Current Mapping");
   isCoordSet = true;
   isValComputed = false;
   isDerComputed = false;

   for (INT aK=0 ; aK< mNbCompVar ; aK++)
   {
      INT aIC = LIC(aK);
      mCompCoord[aIC] = aRealCoord[mMapComp2Real[aIC]];
      // std::cout << "FFFffFFf " << aIC << " " <<mMapComp2Real[aIC] << "\n";
   }

   PostSetCoordCur();
      
}

void cElCompiledFonc::PostSetCoordCur()
{
}


void cElCompiledFonc::ComputeValAndSetIVC()
{
   ELISE_ASSERT(isCoordSet,"No CoordSet Mapping");
   ComputeVal();
   isValComputed = true;
}


REAL cElCompiledFonc::ValBrute(INT aD) const
{
   return mCompCoord[aD];
}



REAL cElCompiledFonc::Val(INT aD) const
{
   ELISE_ASSERT(isValComputed,"No Val Computed");
   return mVal[aD];
}

const std::vector<double> &   cElCompiledFonc::Vals() const
{
   ELISE_ASSERT(isValComputed,"No Val Computed");
   return mVal;
}
const std::vector<double> &   cElCompiledFonc::ValSsVerif() const
{
   return mVal;
}
const std::vector<std::vector<double> > &  cElCompiledFonc::CompDerSsVerif() const
{
   return mCompDer;
}

const std::vector<std::vector<double> > &  cElCompiledFonc::CompDer() const
{
   ELISE_ASSERT(isDerComputed,"No Der Computed");
   return mCompDer;
}
const std::vector<double> &   cElCompiledFonc::CompCoord() const
{
   ELISE_ASSERT(isCoordSet,"No Coord Set");
   return mCompCoord;
}



REAL cElCompiledFonc::Deriv(INT aD,INT aK) const
{
   ELISE_ASSERT(false,"No more Hessian ");
   ELISE_ASSERT(isDerComputed,"No Der Computed");
   AsserNotAlwaysIndexed();
   return 0;

}


REAL cElCompiledFonc::DerSec(INT aD,INT aK1,INT aK2) const
{
   ELISE_ASSERT(false,"No more Hessian ");
   return 0;
}



double * cElCompiledFonc::RequireAdrVarLocFromString(const std::string & aName)
{
  double * aRes  = AdrVarLocFromString(aName);
  if (aRes==0)
  {
     std::cout << aName << "\n";
     ELISE_ASSERT(aRes!=0,"RequireAdrVarLocFromString");
  }
  return aRes;
}



     // Allocation d'objet par nom
cElCompiledFonc::cAutoAddEntry::cAutoAddEntry
(
   const std::string & aStr,
   tAllocObj           anAlloc
)
{
   cElCompiledFonc::AddNewEntryAlloc(aStr,anAlloc);
}

REAL TestMulMat(INT aNbTime, INT aNbVar)
{
    ElTimer aChrono;
    Im2D_REAL8 aMat(aNbVar,aNbVar,0.0);
    Im1D_REAL8 aVec(aNbVar,0.0);

    REAL8 ** aDM = aMat.data();
    REAL8 *  aDV = aVec.data();

    for (INT aKT=0 ; aKT<aNbTime ; aKT++)
        for (INT aKV=0 ; aKV<aNbVar ; aKV++)
	{
		for (INT aX=0 ; aX<aNbVar ; aX++)
		    for (INT aY=0 ; aY<aNbVar ; aY++)
		        aDM[aY][aX] += aDV[aY] * aDV[aX];
	}

    return aChrono.uval();
}

void cElCompiledFonc::SVD_And_AddEqSysSurResol
     (
         bool isCstr,
         const std::vector<INT> & aVIndInit,
	 REAL aPds,
	 REAL *       Pts,
         cGenSysSurResol & aSys,
         cSetEqFormelles & aSet,
	 bool EnPtsCur,
         cParamCalcVarUnkEl * aPCVU
     )
{
   SVD_And_AddEqSysSurResol(isCstr,aVIndInit,MakeVec1(aPds),Pts, aSys,aSet,EnPtsCur,aPCVU);
}


class cBufOneSetEq
{
    public :
       cBufOneSetEq
       (
            // cGenSysSurResol * aSys,
            const std::vector<cSsBloc> *aVSB,
            double *  FullCoeff,
            int aNbTot,
            const std::vector<INT> & aVIndNN ,
            REAL aPds,REAL * aCoeffNN,REAL aB,
            cParamCalcVarUnkEl *
       );
               // aSys.GSSR_AddNewEquation_Indexe

    private :
       cGenSysSurResol *      mSys;
       std::vector<cSsBloc>   mVSB;
       std::vector<double>    mFullCoeff;
       int                    mNbTot;
       std::vector<INT>       mVInd;
       double                 mPds;
       std::vector<double>    mCoeff;
       double                 mB;
       cParamCalcVarUnkEl *   mPCVU;
};



/*
  int aK0AllocSC = -1;
  int aNbBS = 1;
*/
cSsBloc * mSsBlocSpecCond=0;



void cElCompiledFonc::SVD_And_AddEqSysSurResol
     (
         bool isCstr,
         const std::vector<INT> & aVIndInit,
	 const std::vector<double> & aVPds,
	 REAL *       Pts,
         cGenSysSurResol & aSys,
         cSetEqFormelles & aSet,
	 bool EnPtsCur,
         cParamCalcVarUnkEl * aPCVU
     )
{

  int aSzPds = (int)aVPds.size();
  ELISE_ASSERT((aSzPds==1) || (aSzPds==mDimOut),"Taille Pds incohe in cElCompiledFonc::SVD_And_AddEqSysSurResol");

  if (INT(aVIndInit.size())!=mNbCompVar)
  {
     std::cout << "SIZE " << aVIndInit.size() << " " << mNbCompVar << "\n";
     ELISE_ASSERT
     (
       false,
       "cElCompiledFonc::SVD_And_AddEqSysSurReol"
     );
  }


  for (INT aK=0 ; aK< mNbCompVar ; aK++)
       mCompCoord[aK] = Pts[aVIndInit[aK]];

  int aK0AllocSC = -1;
  int aNbBS = 1;
  if (mSsBlocSpecCond)
  {
      aNbBS = mSsBlocSpecCond->Nb();
      ELISE_ASSERT(aNbBS == 3,"Assert Nb in mSsBlocSpecCond");
      for (INT aK=0 ; aK< mNbCompVar ; aK++)
      {
          if (aVIndInit[aK]==mSsBlocSpecCond->I0AbsAlloc())
          {
            // std::cout << " GGgGggg " << mSsBlocSpecCond->I0AbsAlloc() << " " << mSsBlocSpecCond->I0AbsSolve() << "\n";
            ELISE_ASSERT(aK0AllocSC==-1,"Multiple K in mSsBlocSpecCond");
            aK0AllocSC = aK;
          }
      }
      ELISE_ASSERT(aK0AllocSC!=-1,"No K in mSsBlocSpecCond");
      ELISE_ASSERT(aK0AllocSC+aNbBS <= mNbCompVar ,"High K in mSsBlocSpecCond");

      for (int aD=1 ; aD<aNbBS ; aD++)
          ELISE_ASSERT(aVIndInit[aK0AllocSC+aD]==aVIndInit[aK0AllocSC]+aD,"Non continous in mSsBlocSpecCond");

  }

   static bool first = false;
   ElTimer aChrono;
   INT aNb = first ? 1 : 1;
   for (INT aK=0 ; aK< aNb ; aK++)
       ComputeValDeriv();

   isValComputed = true;
   isDerComputed = true;

   bool UseMat = aSys.GSSR_UseEqMatIndexee() && false;

   static std::vector<INT> aVInd; 
   static std::vector<std::vector<double> > 	aVDer;
   static std::vector<double > 	                aVB;

   if (UseMat)
   {
        ELISE_ASSERT(false,"Use mat to implement");
   }
   else
   {
       for (INT aD= 0 ; aD < mDimOut ; aD++)
       {
            if (aD >= int(aVDer.size()))
            {
               aVDer.push_back(std::vector<double>());
               aVB.push_back(0);
            }
            std::vector<double> & aDer = aVDer[aD];
            REAL & aB = aVB[aD];
             
            // double aPdsCur = aVPds[ElMin(aD,aSzPds-1)]; 
            aVInd.clear();
            aDer.clear();
            aB = -mVal[aD];

            bool GotCstr=false; GccUse(GotCstr);
            for (INT aK=0 ; aK< mNbCompVar ; aK++)
            {
                 double aDdk = mCompDer[aD][aK];
                 if (aDdk)  // (PCVU)  Provisoie, + propre de ne pas annuler tte les var tmp
                 {
                     int anInd = aVIndInit[aK];
                    
                     double aValForced;
/*
*/
                    // if (PermutIndex)
                     anInd = aSet.Alloc2Solve(anInd);
                    
                     if(aSys.IsCstrUniv(anInd,aValForced) )
                     {
                         ELISE_ASSERT(!isCstr,"Multiple contrainte univ");
                         GotCstr=true;
                         aB -=  aDdk * aValForced;
                         mCompDer[aD][aK] = 0;
                     }
                     else
                     {
                         if (! EnPtsCur) 
                         {
                             aB +=  aDdk * mCompCoord[aK];
                         }
                         aVInd.push_back(anInd);
                         aDer.push_back(mCompDer[aD][aK]);
                     }
                 }
            }
            // if (GotCstr && isCstr && (aVInd.size() 

/*
            if (isCstr)
            {
               aSys.GSSR_AddContrainteIndexee(aVInd,&(aDer[0]),aB);
            }
            else
            {
               aSys.GSSR_AddNewEquation_Indexe
               ( 
                       &mBlocs,
		       &(mCompDer[aD][0]),
                       (int)aVIndInit.size(),
                       aVInd,
                       aPdsCur,
                       ( ( aDer.size()==0 )?NULL:&(aDer[0]) ),
                       aB ,
                       aPCVU
                );
            }
*/
       }
       if (0) // (mSsBlocSpecCond)
       {
          std::cout << "DDD " << mDimOut << "\n";
          ElMatrix<double> aMCond(aNbBS,aNbBS,0.0);
          for (INT aD= 0 ; aD < mDimOut ; aD++)
          {
             std::cout  << "D1111 "  << "\n";
             for (int aK1=0 ; aK1<aNbBS; aK1++)
             {
                 double aD1 = mCompDer[aD][aK0AllocSC+aK1];
                 std::cout  << aD1 << " ";
                 for (int aK2=0 ; aK2<aNbBS; aK2++)
                 {
                    double aD2 = mCompDer[aD][aK0AllocSC+aK2];
                    aMCond(aK1,aK2) +=  aD1 * aD2;
                 }
             }
             std::cout <<  "\n";
          }
/*
          double aL = aMCond.L2();
          aMCond = gaussj(aMCond);
          aL *=  aMCond.L2();
          aL = sqrt(aL) / ElSquare(aNbBS);
          std::cout << "cccCOND=" << aL << "\n";
*/
       }
       for (INT aD= 0 ; aD < mDimOut ; aD++)
       {
            double aPdsCur = aVPds[ElMin(aD,aSzPds-1)]; 
            std::vector<double> & aDer = aVDer[aD];
            REAL & aB = aVB[aD];
            if (isCstr)
            {
               aSys.GSSR_AddContrainteIndexee(aVInd,&(aDer[0]),aB);
            }
            else
            {
               aSys.GSSR_AddNewEquation_Indexe
               ( 
                       &mBlocs,
		       &(mCompDer[aD][0]),
                       (int)aVIndInit.size(),
                       aVInd,
                       aPdsCur,
                       ( ( aDer.size()==0 )?NULL:&(aDer[0]) ),
                       aB ,
                       aPCVU
                );
            }
       }
   }

}

void cElCompiledFonc::Std_AddEqSysSurResol
     (
         bool isCstr,
	 const std::vector<double> & aVPds,
	 REAL *       Pts,
         cGenSysSurResol & aSys,
         cSetEqFormelles & aSet,
	 bool EnPtsCur,
         cParamCalcVarUnkEl * aPCVU
     )

{


    SVD_And_AddEqSysSurResol(isCstr,mMapComp2Real,aVPds,Pts,aSys,aSet,EnPtsCur,aPCVU);
}

void cElCompiledFonc::Std_AddEqSysSurResol
     (
         bool isCstr,
	 REAL aPds,
	 REAL *       Pts,
         cGenSysSurResol & aSys,
         cSetEqFormelles & aSet,
	 bool EnPtsCur,
         cParamCalcVarUnkEl * aPCVU
     )
{
   return Std_AddEqSysSurResol(isCstr,MakeVec1(aPds),Pts,aSys,aSet,EnPtsCur,aPCVU);
}



void cElCompiledFonc::AddContrainteEqSSR
     (bool contr,REAL aPds,cGenSysSurResol & aSys,bool EnPtsCur)
{
    ELISE_ASSERT
    (
      false,
      "cElCompiledFonc::AddContrainteEqSSR  obsolete"
    );
#if (0)
#endif
}

void cElCompiledFonc::AddDevLimOrd1ToSysSurRes
     (cGenSysSurResol & aSys,REAL aPds,bool EnPtsCur)
{
	AddContrainteEqSSR(false,aPds,aSys,EnPtsCur);
}

void cElCompiledFonc::AddContrainteToSysSurRes
     (cGenSysSurResol & aSys,bool EnPtsCur)
{
	AddContrainteEqSSR(true,-1,aSys,EnPtsCur);
}


void cElCompiledFonc::AddDevLimOrd2ToSysSurRes (L2SysSurResol & aSys,REAL aPds)
{
    ELISE_ASSERT
    (
      false,
      "No more cElCompiledFonc::AddDevLimOrd2ToSysSurRes"
    );
}



/*********************************************************/
/*                                                       */
/*            cElCompileFN                               */
/*                                                       */
/*********************************************************/





void cElCompileFN::SetFile(const std::string & aPostFix, const char * anInclCompl)
{
    if (mFile)
       ElFclose(mFile);

    mNameFile = MMDir() + mNameDir + mNameClass + "." + aPostFix;
    mFile = ElFopen(mNameFile.c_str(),"w");
    if (mFile==0)
    {
        std::cout << "FOR FILE=[" << mNameFile << "]\n";
        ELISE_ASSERT(false,"Cannot open FILE in cElCompileFN");
    }


    mNameTagInclude = "_" + mNameClass + "_" + aPostFix + "_";

   (*this)<< "// File Automatically generated by eLiSe\n";

    // (*this)<< "#ifndef " << mNameTagInclude << "\n";
    // (*this)<< "#define " << mNameTagInclude << "\n\n";

    // cout << " --SUPPRIME -- #ifndef " << mNameTagInclude << "\n";
    // cout << " --SUPPRIME -- #define " << mNameTagInclude << "\n\n";

   (*this)<< "#include \"StdAfx.h\"\n";

    if (anInclCompl)
    {
       (*this)<< "#include \"" <<  mNameClass << "." << anInclCompl << "\"\n";
    }

    (*this)<< "\n\n";


}

void cElCompileFN:: CloseFile()
{
    if (mFile)
    {
       // (*this)<< "#endif // " << mNameTagInclude << "\n";
       // cout << "-- SUPPRIME --  #endif // " << mNameTagInclude << "\n";
       ElFclose(mFile);
       mFile = 0;
    }
}

cElCompileFN::cElCompileFN
(
      const std::string &             aNameDir,
      const std::string &             aNameClass,
      const cIncListInterv &          aListInterv
) :
     mFile          (0),
     mNamesLoc      (new cECFN_SetString),
     mDicSymb       ( new  cDico_SymbFN),
     // mNVMax         (aListInterv.Surf()),
     mNVMax         (aListInterv.I1Max()),
     mNameVarNum    ("mCompCoord"),
     mPrefVarLoc    ("mLoc"),
     mNameDir       (aNameDir),
     mNameClass     (aNameClass),
     mListInterv    (aListInterv)
{
}

cElCompileFN::~cElCompileFN()
{
   CloseFile();
   delete mNamesLoc;
   delete mDicSymb;
}

cElCompileFN &  cElCompileFN::operator << (const char * aStr)
{
  if (mFile)
     fprintf(mFile,"%s",aStr);
  return *this;
}

cElCompileFN &  cElCompileFN::operator << (const std::string & aStr)
{
  if (mFile)
     fprintf(mFile,"%s",aStr.c_str());
  return *this;
}

cElCompileFN &  cElCompileFN::operator << (const INT & anInt)
{
  if (mFile)
     fprintf(mFile,"%d",anInt);
  return *this;
}

cElCompileFN &  cElCompileFN::operator << (const double & aNum)
{
  if (mFile)
     fprintf(mFile,"%f",aNum);
  return *this;
}

cElCompileFN &  cElCompileFN::operator << (Fonc_Num  & aFonc )
{
  if ((mFile==0)||aFonc.IsVarSpec())
  {
     aFonc.compile(*this);
  }
  else
  {
     mDicSymb->PutFonc(aFonc,*this);
  }
  return *this;
}

void cElCompileFN::PutVarNum(INT aK)
{
    ELISE_ASSERT(aK<mNVMax,"Bad Var Number in cElCompileFN::PutVarNum");
    (*this) << mNameVarNum << "[" << aK << "]";
}

std::string   cElCompileFN::NameVarLoc(const std::string & aStr)
{
    return mPrefVarLoc + aStr;
}

void  cElCompileFN::PutVarLoc(cVarSpec aVar)
{
    mNamesLoc->insert(aVar);
   // (*this) << mPrefVarLoc << aStr ;
   (*this) << NameVarLoc(aVar.Name());
}


void cElCompileFN::MakeFileCpp(std::vector<Fonc_Num> vFoncs,bool SpecFCUV)
{

   SetFile("cpp","h");

   int aDimOut = (int)vFoncs.size();


// Constructeur : 

   (*this) << mNameClass << "::" << mNameClass << "():\n";
   (*this) << "    cElCompiledFonc(" <<  aDimOut << ")\n";
   (*this) << "{\n";


   const cMapIncInterv & aMap =  mListInterv.Map();
   for (tCSetIII itCur = aMap.begin() ; itCur != aMap.end() ; itCur++)
   {
      (*this) << "   AddIntRef (cIncIntervale("
              <<  "\"" <<  itCur->Id()  <<  "\"" 
              <<  "," << itCur->I0Alloc()
              <<  "," << itCur->I1Alloc()
              << "));\n";
   }

   (*this) << "   Close(false);\n";

   (*this) << "}\n";
   (*this) << "\n\n\n";


   MakeFonc(vFoncs,0,SpecFCUV);
   MakeFonc(vFoncs,1,SpecFCUV);
   MakeFonc(vFoncs,2,SpecFCUV);

// Calcul des Fonction SetVar
   (*this) << "\n";
   for (cECFN_SetString::const_iterator it = mNamesLoc->begin(); it!=mNamesLoc->end() ; it++)
   {
       (*this)  << "void " << mNameClass << "::Set"<< it->Name() << "(double aVal){ "
                << NameVarLoc(it->Name()) << " = aVal;}\n";
   }
   (*this) << "\n\n\n";


// Fonction AdrVarLocFromString : 
   (*this) << "double * " << mNameClass << "::AdrVarLocFromString(const std::string & aName)\n";
   (*this) << "{\n";
   {
   for (cECFN_SetString::const_iterator it = mNamesLoc->begin(); it!=mNamesLoc->end() ; it++)
   {
       (*this)   << "   if (aName == \"" 
                 << (it->Name()) << "\") return & " 
                 << NameVarLoc(it->Name()) 
                 << ";\n";
   }
   }
   (*this) << "   return 0;\n";
   (*this) << "}\n\n\n";


   (*this)  << "cElCompiledFonc::cAutoAddEntry "  
	    << mNameClass << "::mTheAuto"
	    << "(\"" << mNameClass << "\"," 
	    << mNameClass << "::Alloc);\n\n\n";

   (*this)  << "cElCompiledFonc *  " << mNameClass << "::Alloc()\n";
   (*this)  << "{";
   (*this)  << "  return new " << mNameClass << "();\n";
   (*this)  << "}\n\n\n";

    
   CloseFile();
}


void cElCompileFN::MakeFileH(bool SpecFCUV)
{
   SetFile("h",0);

   (*this) << "class " << mNameClass << ": public cElCompiledFonc\n";
   (*this) << "{\n";
   (*this) << "   public :\n\n";
   (*this) << "      " << mNameClass << "();\n";
   (*this) << "      void ComputeVal();\n";
   (*this) << "      void ComputeValDeriv();\n";
   (*this) << "      void ComputeValDerivHessian();\n";
   (*this) << "      double * AdrVarLocFromString(const std::string &);\n";
   for (cECFN_SetString::const_iterator it = mNamesLoc->begin(); it!=mNamesLoc->end() ; it++)
   {
       (*this)  << "      void Set"<<  it->Name() << "(double);\n";
   }


   (*this) << "\n\n";

   (*this)  << "      static cAutoAddEntry  mTheAuto;\n";
   (*this)  << "      static cElCompiledFonc *  Alloc();\n";
   (*this) << "   private :\n\n";

   {
   for (cECFN_SetString::const_iterator it = mNamesLoc->begin(); it!=mNamesLoc->end() ; it++)
   {
       (*this)  << "      double "<<NameVarLoc(it->Name()) <<";\n";
   }
   }

   (*this) << "};\n";

   CloseFile();
}

extern bool FnumCoorUseCsteVal;

Fonc_Num SimplifyFCUV(Fonc_Num aF)
{
   if (FnumCoorUseCsteVal) return aF.Simplify();
   return aF;
}


// On met d'abord les fonction pour la derivees, ensuite les fonctions pour la valeur

#define MAKE_DER_SEC false
void cElCompileFN::MakeFonc(std::vector<Fonc_Num> vFoncs,INT aDegDeriv,bool UseAccel) // 0, fonc , 1 deriv, 2 hessian
{
   FnumCoorUseCsteVal = UseAccel;
   int aDimOut = (int)vFoncs.size();
// std::cout << "DEGRE " << aDegDeriv << "\n";
    if (! MAKE_DER_SEC && (aDegDeriv==2))
    {
        (*this) << "void " << mNameClass << "::ComputeValDerivHessian()\n";
        (*this) << "{\n";
        (*this) << "  ELISE_ASSERT(false,\"Foncteur " 
                << mNameClass << " Has no Der Sec\");\n";
        (*this) << "}\n";
        return;
    }


    FILE * aSauvFile = mFile;
    mFile = 0;

    delete mDicSymb;
    mDicSymb     = new  cDico_SymbFN;

    int aK0 = 0 ;
    for (INT aK= 0  ; aK<aDimOut ; aK++)
    {
         Fonc_Num aF0 =  SimplifyFCUV(vFoncs[aK]);  
         (*this) << aF0;

         if (aDegDeriv >= 1)
         {
            for (INT aD=0 ; aD<mNVMax ; aD++)
            {
	        Fonc_Num aFD =  SimplifyFCUV(vFoncs[aK].deriv(aD));
                (*this) << aFD;
            }
         }
         if (aDegDeriv >= 2)
         {
            for (INT aD1=0 ; aD1<mNVMax ; aD1++)
            {
	        Fonc_Num aFD1 =  SimplifyFCUV(vFoncs[aK].deriv(aD1));
                for (INT aD2=0 ; aD2<= aD1 ; aD2++)
                {
	            Fonc_Num aFD1D2 = SimplifyFCUV(aFD1.deriv(aD2));
                    (*this) << aFD1D2;
                }
            }
         }
    }
    mFile = aSauvFile;


// Calcul de la valeur

   (*this) << "void " << mNameClass << "::ComputeVal";
   if (aDegDeriv>=1) (*this) << "Deriv";
   if (aDegDeriv>=2) (*this) << "Hessian";
   (*this) << "()\n";

   (*this) << "{\n";
   mDicSymb->PutSymbs(*this);

   for (INT aK=0; aK<aDimOut ; aK++)
   {
       Fonc_Num aF0 =  SimplifyFCUV(vFoncs[aK]);
       (*this) << "  mVal["<<aK-aK0<<"] = " << aF0 << ";\n\n";

        // Calcul des derivees 
       if (aDegDeriv>=1)
       {
           for (INT aD = 0 ; aD<mNVMax ; aD++)
           {
                Fonc_Num aDer =  SimplifyFCUV(vFoncs[aK].deriv(aD) );
                (*this) << "  mCompDer[" <<aK-aK0 << "][" << aD << "] = " << aDer << ";\n";
           }
       }

        // Calcul des derivees Seconde
       if (aDegDeriv>=2)
       {
           for (INT aD1 = 0 ; aD1<mNVMax ; aD1++)
           {
                Fonc_Num aFD1 =   SimplifyFCUV(vFoncs[aK].deriv(aD1));
                for (INT aD2=0 ; aD2<= aD1 ; aD2++)
                {
	           Fonc_Num aFD1D2 = SimplifyFCUV(aFD1.deriv(aD2));
                   (*this) 
                       << "  mCompHessian[" << aK<< "][" << aD1 << "]["<< aD2 << "]= " 
                       << "  mCompHessian[" << aK<< "][" << aD2 << "]["<< aD1 << "]= " 
                       << aFD1D2 << ";\n";
                }
           }
       }
   }
   (*this) << "}\n\n\n";
   FnumCoorUseCsteVal = false;
}


void cElCompileFN::DoEverything
     (
             const std::string &             aDir,
             const std::string &             aName,
	     std::vector<Fonc_Num>           vFoncs,
             const cIncListInterv &          aListInterv,
             bool                            SpecFCUV
     )
{
   FnumCoorUseCsteVal = SpecFCUV;
   ELISE_ASSERT(aListInterv.IsConnexe0N(),"Bad Interv in cElCompileFN::DoEverything");

   cElCompileFN aECFN(aDir,aName,aListInterv);

 // On parcourt une premiere fois  pour generer les variables, a toutes fins utiles
   for (INT aK=0; aK<INT(vFoncs.size()) ; aK++)
   {
       Fonc_Num aF = SimplifyFCUV( vFoncs[aK]);
       aECFN << aF ;
       INT aNb = aListInterv.Surf();
       for (INT aD=0 ; aD<aNb ; aD++)
       {
	    Fonc_Num aFD = vFoncs[aK].deriv(aD);
            aECFN << aFD;
       }
   }

   aECFN.MakeFileCpp(vFoncs,SpecFCUV);
   aECFN.MakeFileH(SpecFCUV);
   FnumCoorUseCsteVal = false;
}

void cElCompileFN::DoEverything
     (
             const std::string &             aDir,
             const std::string &             aName,
	     Fonc_Num                        aFonc,
             const cIncListInterv &          aListInterv
     )
{
	std::vector<Fonc_Num> aVec;
	aVec.push_back(aFonc);
	DoEverything(aDir,aName,aVec,aListInterv);
}

void cElCompileFN::AddToDict(Fonc_Num aF)
{
    if (!mFile)
       mDicSymb->AddF(aF);
}

bool cElCompileFN:: SqueezComp(Fonc_Num aF)
{
   if ( mFile) 
      return false;
   return  mDicSymb->SqueezComp(aF);
}



/**************************************************************/
/*                                                            */
/*              cDynFoncteur                                  */
/*                                                            */
/**************************************************************/




   //=====================


class cDynFoncteur : public cElCompiledFonc
{
    public :

          cDynFoncteur
          (
                  const cIncListInterv &  aListInterv,
                  const cECFN_SetString &,
                  Fonc_Num f
          );

    private  :

         double * AdrVarLocFromString(const std::string &); 
         
         void ComputeVal() ;
         void ComputeValDeriv() ;
         void ComputeValDerivHessian() ;
         void PostSetCoordCur();

         cECFN_SetString         mScVS;
         Fonc_Num                mFoncForm;
         std::vector<Fonc_Num>   mDerForm;
         PtsKD                   mPts;
};


void cDynFoncteur::ComputeVal()
{
   ELISE_ASSERT(mDimOut==1,"Dim !=1,DynFoncteur");
   mVal[0] = mFoncForm.ValFonc(mPts);
}


void cDynFoncteur::ComputeValDeriv()
{
   ELISE_ASSERT(mDimOut==1,"Dim !=1,DynFoncteur");
   mVal[0] = mFoncForm.ValFonc(mPts);
   for (INT aD = 0 ; aD<mNbCompVar ; aD++)
   {
       INT aIC = LIC(aD);
       mCompDer[0][aIC] = mDerForm[aD].ValFonc(mPts);
   }
}

void cDynFoncteur::ComputeValDerivHessian() 
{
   ELISE_ASSERT(false,"No More Hess");
}



void cDynFoncteur::PostSetCoordCur()
{
   for (INT aD=0 ; aD<mNbCompVar ; aD++)
   {
       INT aIC = LIC(aD);
       mPts(aIC) = mCompCoord[aIC] ;
   }
}


cDynFoncteur::cDynFoncteur
(
     const cIncListInterv &     aListInterv,
     const cECFN_SetString &    aScVS,
     Fonc_Num                   aFonc
)  :
   cElCompiledFonc(1),
   mScVS       (aScVS),
   mFoncForm   (aFonc),
   mPts        (aListInterv.I1Max())
{
   // ELISE_ASSERT(aListInterv.IsConnexe0N(),"Bad Interv in cDynFoncteur::cDynFoncteur");
   const cMapIncInterv & aMap =  aListInterv.Map();
   for (tCSetIII itCur = aMap.begin() ; itCur != aMap.end() ; itCur++)
        AddIntRef (*itCur);
   Close(true);

   for (INT aD=0 ; aD<mNbCompVar ; aD++)
   {
       INT aIC = LIC(aD);
       mDerForm.push_back(aFonc.deriv(aIC));
   }
       
}

double * cDynFoncteur::AdrVarLocFromString(const std::string & aName)
{
   for (cECFN_SetString::const_iterator it = mScVS.begin(); it!=mScVS.end() ; it++)
       if (it->Name() == aName)
          return it->AdrVal();

   return 0;
}


cElCompiledFonc * cElCompileFN::DynamicAlloc
                  (
                    const cIncListInterv &  aListInterv,
                    Fonc_Num                aFonc
                  )
{
	/*
   ELISE_ASSERT
   (
      aListInterv.IsConnexe0N(),
      "Bad Interv in cElCompileFN::DoEverything"
   );
   */

   cElCompileFN aECFN("","",aListInterv);
   aECFN << aFonc;

   return new cDynFoncteur(aListInterv,*aECFN.mNamesLoc,aFonc);
}

cElCompiledFonc * cElCompiledFonc::DynamicAlloc
                  (
                    const cIncListInterv &  aListInterv,
                    Fonc_Num                aFonc
                  )

{
   return cElCompileFN::DynamicAlloc(aListInterv,aFonc);
}


void  cDico_SymbFN::PutFonc(Fonc_Num  aF,cElCompileFN & aComp)
{
   tIter anIt = mDic.find(aF);
   if (anIt==mDic.end())
   {
       aF.compile(aComp);
       return;
   }

   cCelDico_SFN & aCel = anIt->second;

   if ( aCel.mSymbPut)
      aComp << aCel.NameSymb();
   else 
       aF.compile(aComp);
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
