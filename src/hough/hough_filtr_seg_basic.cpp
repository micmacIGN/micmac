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

#include "hough_include.h"

#define NoTemplateOperatorVirgule




class StateEHFS_PrgDyn;
class SetState_EHFS_PD;
class EHFS_PrgDyn;
class ElHoughFiltSeg;



/************************************************************************/
/************************************************************************/
/****                                                                 ***/
/****                                                                 ***/
/****             DECLARATION DES CLASSES                             ***/
/****                                                                 ***/
/****                                                                 ***/
/************************************************************************/
/************************************************************************/


    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 StateEHFS_PrgDyn                                          */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/

class StateEHFS_PrgDyn
{
     public :
       typedef const StateEHFS_PrgDyn *     tPtrPred;
       typedef ElSTDNS list<tPtrPred>       tLPred;
       typedef tLPred::iterator             tItPred;

       StateEHFS_PrgDyn(bool IsSeg,INT anX);
       void ComputeGain(ElHoughFiltSeg &,bool IsExtre,bool IsLast);
       void AddPred(tPtrPred);
       void AddPred(const ElSTDNS list<StateEHFS_PrgDyn>&);
       void SetPred(const ElSTDNS list<StateEHFS_PrgDyn> &);
       const StateEHFS_PrgDyn * BestPred () const;

       bool GainGlobInf (const StateEHFS_PrgDyn & ) const;
       bool IsSeg() const;
       INT  X() const;
	   REAL	GainElem() const { return mGainElem; }
	   REAL	GainGlob() const { return mGainGlob; }
	   tLPred	Preds() const { return mPreds; }

       friend ostream & operator << (ostream & Out,const StateEHFS_PrgDyn &);

     private :

       INT             mX;
       REAL            mGainGlob;
       tLPred          mPreds;
       tPtrPred        mBestPred;
       REAL            mGainElem;
       bool            mIsSeg;

};


    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 SetState_EHFS_PD                                          */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/


class SetState_EHFS_PD
{
     public :
          SetState_EHFS_PD(INT x,bool WithSegHyp);
          void AddPred(SetState_EHFS_PD & Pred);
          void SetPred(SetState_EHFS_PD & Pred);
          void ComputeGain(ElHoughFiltSeg &,bool IsExtre,bool IsLast);
          StateEHFS_PrgDyn * Best();

       friend ostream & operator << (ostream & Out,const SetState_EHFS_PD &);

	   ElSTDNS list<StateEHFS_PrgDyn>::const_iterator stateBegin() const { return mStates.begin(); }
	   ElSTDNS list<StateEHFS_PrgDyn>::const_iterator stateEnd() const { return mStates.end(); }

     private :
         INT mX;
         ElSTDNS list<StateEHFS_PrgDyn>   mStates;
};

    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 EHFS_PrgDyn                                               */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/

class EHFS_PrgDyn
{
    public :
       EHFS_PrgDyn(INT NbMax);
       void  ComputeBestChem
             (
                  std::vector<ElSTDNS  pair<INT,INT> > & res,
                  ElHoughFiltSeg &,
                  INT Nb
             );

    private :
        //INT mNbMax;
        typedef std::vector<SetState_EHFS_PD>  tChem;
        void ComputeGain(ElHoughFiltSeg &,INT Nb);

        SetState_EHFS_PD   mSfront;
        SetState_EHFS_PD   mSback;
        tChem              mChem;                
      
};


/************************************************************************/
/************************************************************************/
/****                                                                 ***/
/****                                                                 ***/
/****             DEFINITION DES CLASSES                              ***/
/****                                                                 ***/
/****                                                                 ***/
/************************************************************************/
/************************************************************************/


    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 StateEHFS_PrgDyn                                          */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/



ostream & operator << (ostream & Out,const StateEHFS_PrgDyn &St)
{
   Out 
       << "[X=" << St.X() 
       << ";IsSeg=" << St.IsSeg() 
       << ";GainEl=" << (INT)(100*St.GainElem())
       << ";GainGlob=" << (INT)(100*St.GainGlob())
       << "Nb Pred " << (unsigned int) St.Preds().size()
       << "]";
   return Out;
}

StateEHFS_PrgDyn::StateEHFS_PrgDyn(bool isSeg,INT anX) :
    mX        (anX),
    mGainGlob (0),
    mPreds    (),
    mBestPred (0),
    mGainElem (0),
    mIsSeg    (isSeg)
{
}

bool StateEHFS_PrgDyn::IsSeg() const
{
   return mIsSeg;
}

INT StateEHFS_PrgDyn::X() const
{
   return mX;
}

bool StateEHFS_PrgDyn::GainGlobInf (const StateEHFS_PrgDyn & aState) const
{
    return mGainGlob < aState.mGainGlob;
}

const StateEHFS_PrgDyn * StateEHFS_PrgDyn::BestPred () const
{
   return mBestPred;
}

void StateEHFS_PrgDyn::AddPred(StateEHFS_PrgDyn::tPtrPred aPred)
{
    mPreds.push_back(aPred);
}

void StateEHFS_PrgDyn::AddPred(const ElSTDNS list<StateEHFS_PrgDyn> &l)
{
    for 
    (
        ElSTDNS list<StateEHFS_PrgDyn>::const_iterator it=l.begin() ;
        it!=l.end();
        it++
    )
        AddPred(&(*it));
}

void StateEHFS_PrgDyn::SetPred(const ElSTDNS list<StateEHFS_PrgDyn> &l)
{
    ELISE_ASSERT(l.size()==mPreds.size(),"StateEHFS_PrgDyn::SetPred");
    ElSTDNS list<StateEHFS_PrgDyn>::const_iterator it=l.begin() ;
    tItPred  it2= mPreds.begin();

    for ( ; it!=l.end() ; it++,it2++)
        *(it2) = &(*it);
}




void StateEHFS_PrgDyn::ComputeGain(ElHoughFiltSeg & Ehfs,bool IsExtre,bool IsLast)
{
   mGainElem = IsLast ? Ehfs.CostNeutre() : Ehfs.CostState(mIsSeg,mX);

   mGainGlob = 0;
   mBestPred = 0;

   for (tItPred it=mPreds.begin(); it!=mPreds.end() ; it++)
   {
       const StateEHFS_PrgDyn * aPred = *it;
       REAL gain = aPred->mGainGlob + mGainElem;
       if (
               (mIsSeg != aPred->mIsSeg)
             &&((!IsExtre) || (! Ehfs.ExtrFree()))
          )
          gain -= Ehfs.CostChange(mIsSeg,mX,aPred->mX)/Ehfs.Step();
       if ((!mBestPred) || (gain>mGainGlob))
       {
           mGainGlob = gain;
           mBestPred = aPred;
       }
   }
}

    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 SetState_EHFS_PD                                          */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/


SetState_EHFS_PD::SetState_EHFS_PD(INT x,bool WithSegHyp) :
    mX (x)
{
    mStates.push_back(StateEHFS_PrgDyn(false,x));
    if (WithSegHyp)
        mStates.push_back(StateEHFS_PrgDyn(true,x));

} 


ostream & operator << (ostream & Out,const SetState_EHFS_PD & Set)
{
   Out << "#{";
   for 
   (
       ElSTDNS list<StateEHFS_PrgDyn>::const_iterator  it = Set.stateBegin();
       it != Set.stateEnd();
       it++
   )
      Out << (*it);
   Out << "#}";

   return Out;
}

StateEHFS_PrgDyn * SetState_EHFS_PD::Best()
{
   ELISE_ASSERT(!mStates.empty()," SetState_EHFS_PD::Best");
   StateEHFS_PrgDyn * res = &(*mStates.begin());
   for 
   (
       ElSTDNS list<StateEHFS_PrgDyn>::iterator  it = mStates.begin();
       it != mStates.end();
       it++
   )
      if (res->GainGlobInf(*it))
         res = &(*it);

   return res;
}

void SetState_EHFS_PD::AddPred(SetState_EHFS_PD & Pred)
{
    for 
    (
       ElSTDNS list<StateEHFS_PrgDyn>::iterator it=mStates.begin();
       it != mStates.end();
       it++
    )
       it->AddPred(Pred.mStates);
        
}


void SetState_EHFS_PD::SetPred(SetState_EHFS_PD & Pred)
{
    for 
    (
       ElSTDNS list<StateEHFS_PrgDyn>::iterator it=mStates.begin();
       it != mStates.end();
       it++
    )
       it->SetPred(Pred.mStates);
}

void  SetState_EHFS_PD::ComputeGain(ElHoughFiltSeg & Ehfs,bool IsExtre,bool IsLast)
{
    for 
    (
       ElSTDNS list<StateEHFS_PrgDyn>::iterator it=mStates.begin();
       it != mStates.end();
       it++
    )
      it->ComputeGain(Ehfs,IsExtre,IsLast);
}

    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 EHFS_PrgDyn                                               */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/

EHFS_PrgDyn::EHFS_PrgDyn (INT NbMax) :
    mSfront(-1,false),
    mSback (NbMax+1,false)
{
    mChem.reserve(NbMax+1);
    for (INT k=0; k<=NbMax ; k++)
    {
        mChem.push_back(SetState_EHFS_PD(k,true));
        if (k>0)
           mChem[k].AddPred(mChem[k-1]);
    }
    mChem.front().AddPred(mSfront);
    mSback.AddPred(mChem.back());
}

        

void EHFS_PrgDyn::ComputeGain
     (
         ElHoughFiltSeg &  Ehfs,
         INT Nb
     )
{
    mSback.SetPred(mChem[Nb]);
    for (INT k=0; k<=Nb ; k++)
    {
       mChem[k].ComputeGain(Ehfs,k==0,false);
    }
    mSback.ComputeGain(Ehfs,true,true);
}

void  EHFS_PrgDyn::ComputeBestChem
      (
           std::vector<ElSTDNS pair<INT,INT> > & res,
           ElHoughFiltSeg &  Ehfs,
           INT Nb
      )
{
    res.clear();
    ComputeGain(Ehfs,Nb);
    INT xDeb = -100;
    
    const StateEHFS_PrgDyn * aState = mSback.Best();
    while (aState)
    {
        const StateEHFS_PrgDyn *nextState = aState->BestPred();
        if (nextState) 
        {
            if ((!aState->IsSeg()) && nextState->IsSeg())
               xDeb = nextState->X();

            if ((aState->IsSeg()) && (!nextState->IsSeg()))
            {
               ELISE_ASSERT(xDeb!=-100, "EHFS_PrgDyn::ComputeBestChem");
               res.push_back(ElSTDNS pair<INT,INT>(xDeb,aState->X()));
               xDeb = -100;
            }
        }

        aState = nextState;
    }
    ELISE_ASSERT(xDeb==-100, "EHFS_PrgDyn::ComputeBestChem");
}


    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/
    /*                                                                           */
    /*                 ElHoughFiltSeg                                            */
    /*                                                                           */
    /*+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+*/


   // MACRO DEFINITION POUR  CONTOURNER LES membre-template


             // VerifSize

#define DEFINE_EHFS_VerifSize_IM2(Type,TypeBase)\
void ElHoughFiltSeg::VerifSize(Im2D<Type,TypeBase> anIm)\
{\
     ELISE_ASSERT\
     (\
         (anIm.tx()>=mNbX)&&(anIm.ty()>=SzMax().y),\
         "Bad Size in ElHoughFiltSeg"\
     );\
}


             // VerifSize

#define DEFINE_EHFS_VerifSize_IM1(Type,TypeBase)\
void ElHoughFiltSeg::VerifSize(Im1D<Type,TypeBase> anIm)\
{\
     ELISE_ASSERT\
     (\
         (anIm.tx()>=mNbX),\
         "Bad Size in ElHoughFiltSeg"\
     );\
}


             // MakeIm

#define DEFINE_EHSF_MakeIm(Type,TypeBase)\
void ElHoughFiltSeg::MakeIm(Im2D<Type,TypeBase> Res,Im2D<Type,TypeBase> InPut,Type def)\
{\
     VerifSize(Res);\
     Res.raz();\
\
     TIm2D<Type,TypeBase> mTim(InPut);\
     Type ** data = Res.data();\
\
\
     for (INT y=0; y<= 2* mNbYMax ; y++)\
         for (INT x=0; x<=mNbX ; x++)\
         {\
             Pt2dr PtAbs = Loc2Abs(Pt2dr(x,y));\
             ElPFixed<NbBits> pt (PtAbs);\
\
             data[y][x] = TImGet<Type,TypeBase,NbBits>::geti(mTim,pt,def);\
         }\
}


      // MACRO INSTANTIATION

DEFINE_EHFS_VerifSize_IM2(U_INT1,INT)
DEFINE_EHFS_VerifSize_IM2(INT1,INT)

DEFINE_EHFS_VerifSize_IM1(U_INT1,INT)

DEFINE_EHSF_MakeIm(U_INT1,INT)
DEFINE_EHSF_MakeIm(INT1,INT)

    //   END MACRO




ElHoughFiltSeg::ElHoughFiltSeg
(
      REAL Step,
      REAL WidthMax,
      REAL LengthMax,
      Pt2di Box
) :
  mStep        (Step),
  mWidthMax    (WidthMax),
  mLengthMax   (LengthMax),
  mNbXMax      (round_up(LengthMax/Step)),
  mNbYMax      (round_up(WidthMax/Step)),
  mPrgDyn      (new EHFS_PrgDyn(mNbXMax+1)),
  mBox         (Pt2di(0,0),Box)
{
}

ElHoughFiltSeg::~ElHoughFiltSeg() {}




void ElHoughFiltSeg::SetSeg(const SegComp & aSeg)
{
   mLength = ElMin(aSeg.length(),mLengthMax);
   mNbX    = ElMin(mNbXMax,round_up(mLength/mStep));



   mInvTgtSeg =  Pt2dr(1.0,0.0) / aSeg.tangente();
   mCS2Loc = Pt2dr(1.0,0.0) / (aSeg.tangente() *mStep);
   mP02Loc = Pt2dr(0,mNbYMax)-aSeg.p0()*mCS2Loc;

   mCS2Abs =  Pt2dr(1.0,0.0) / mCS2Loc;
   mP02Abs =  -mP02Loc / mCS2Loc;

   UpdateSeg();
}




Pt2di ElHoughFiltSeg::SzMax()  const
{
    return Pt2di(mNbXMax+1,2*mNbYMax+1);
}
Pt2di ElHoughFiltSeg::SzCur() const
{
    return Pt2di(mNbX+1,2*mNbYMax+1);
}


void ElHoughFiltSeg::GenPrgDynGet(std::vector<Seg2d> & Segs,REAL dMin)
{
    mPrgDyn->ComputeBestChem(mVPairI,*this,mNbX);

    Segs.clear();
    for 
    (
       ElSTDNS vector<ElSTDNS pair<INT,INT> >::iterator itP=mVPairI.begin();
       itP!=mVPairI.end();
       itP++
    )
    {
        Pt2dr p0 = Loc2Abs(Pt2dr(itP->first ,mNbYMax));
        Pt2dr p1 = Loc2Abs(Pt2dr(itP->second,mNbYMax));
        if (euclid(p0,p1)> dMin)
            Segs.push_back(Seg2d(p0,p1)); 
    }
}




Seg2d ElHoughFiltSeg::OneExtendSeg
      (
        bool  &       Ok,
        SegComp       s0,
        Im2DGen &     Im
      )
{


      Ok = false; 
      REAL delta = ElMax(10+AverageCostChange()*3,5+s0.length()*0.25);
      Seg2d  s1 (
                 s0.p0()-s0.tangente() * delta,
                 s0.p1()+s0.tangente() * delta
             );
      s1 = s1.clip(Im.ImBox2d(4));

      if (s1.empty())
         return s0;


      SetSeg(s1);
      GenPrgDynGet(mVSegsExt,s0.length());

      REAL  RecouMax = s0.length()*0.66;
      Seg2d * sMax =  0;

      for 
      (
           ElSTDNS vector<Seg2d>::iterator itS = mVSegsExt.begin();
           itS != mVSegsExt.end();
           itS++
      )
      {
           REAL rec = s0.recouvrement_seg(*itS);
           if (rec > RecouMax)
           {
               RecouMax = rec;
               sMax = &(*itS);
           }
      }

      if (! sMax) 
         return s0;
      s1 = *sMax;


      REAL d1 = euclid(s1.p0(),s1.p1());
      if (d1 <= s0.length())
         return s0;
      INT iD1 = ElMax(1,round_ni(d1));

      REAL ScOpt;
      s1 = Im.OptimizeSegTournantSomIm(ScOpt,s1,iD1,1.0,0.1);

      Ok = true;
      return s1;
}


Seg2d ElHoughFiltSeg::ExtendSeg
      (
        SegComp       s0,
        REAL          DeltaMin,
        Im2DGen &     Im
      )
{
     Seg2d sclip = s0.clip(Im.ImBox2d(4));
     if (sclip.empty())
        return s0;
     s0 = sclip;
     REAL ScOpt;
     INT iL = ElMax(1,round_ni(s0.length()));
     s0 = Im.OptimizeSegTournantSomIm(ScOpt,s0,iL,1.0,0.1);

      bool Ok=true;

      while (Ok)
      {
          SegComp s1 = OneExtendSeg(Ok,s0,Im);
          if (Ok)
          {
              if (s1.length()<=s0.length()+DeltaMin)
                 Ok = false;
              s0 = s1;
          }
      }
      return s0;
}

INT ElHoughFiltSeg::YCentreCur() const
{
   return SzCur().y/2;
}


bool ElHoughFiltSeg::ExtrFree()   {return false;}
REAL ElHoughFiltSeg::CostNeutre() {return 0.5;}
REAL ElHoughFiltSeg::CostChange(bool,INT,INT)
{
   return AverageCostChange();
}

void ElHoughFiltSeg::ExtendImage_proj(Im2D_U_INT1 Im,INT Delta)
{
   VerifSize(Im);
   U_INT1 ** Data = Im.data();

   ElSetMin(Delta,mNbXMax-mNbX);

   for (INT y=0; y<=2*mNbYMax ; y++)
       set_cste(Data[y]+mNbX,Data[y][mNbX-1],Delta);
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
