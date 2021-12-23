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

#define NoTemplateOperatorVirgule
#define NoSimpleTemplateOperatorVirgule


#include "StdAfx.h"
#include "../src/uti_phgrm/Apero/cCameraRPC.h"

extern bool ERupnik_MM();

bool DebugOFPA = false;
int aCPTOkOFA = 0;
int aCPTNotOkOFA = 0;
bool BugFE = false;
bool BugAZL = false;

static void ToSepList
            (
                    ElSTDNS list<Pt3dr> & PR3 ,
                    ElSTDNS list<Pt2dr> & PF2 ,
                    const ElSTDNS list<Appar23> & P23
            )
{
    PR3.clear();
    PF2.clear();

    for
    (
          ElSTDNS list<Appar23>::const_iterator It23 = P23.begin();
          It23 != P23.end();
          It23++
    )
    {
        PR3.push_back(It23->pter);
        PF2.push_back(It23->pim);
    }
}




/*************************************************/
/*                                               */
/*    Appar23                                    */
/*                                               */
/*************************************************/

Appar23::Appar23 (Pt2dr PIM,Pt3dr PTER,int aNum) :
   pim   (PIM),
   pter  (PTER),
   mNum  (aNum)
{
}



Appar23  BarryImTer(const std::list<Appar23> & aLAp)
{
   ELISE_ASSERT(!aLAp.empty(),"BarryImTer : liste vide !! ");
   Pt3dr aPTer(0,0,0);
   Pt2dr aPIm(0,0);
   double aS=0;
   for
   (
       std::list<Appar23>::const_iterator itAp=aLAp.begin();
       itAp!=aLAp.end();
       itAp++
   )
   {
      aS++;
      aPTer = aPTer + itAp->pter;
      aPIm  = aPIm  + itAp->pim;
   }

   return Appar23(aPIm/aS,aPTer/aS);
}

void InvY(std::list<Appar23> & aLAp,Pt2dr aSzIm,bool InvX)
{
   for
   (
       std::list<Appar23>::iterator itAp=aLAp.begin();
       itAp!=aLAp.end();
       itAp++
   )
   {
       itAp->pim.y = aSzIm.y - itAp->pim.y;
       if (InvX)
          itAp->pim.x = aSzIm.x - itAp->pim.x;
   }
}

/*************************************************/
/*                                               */
/*    cNupletPtsHomologues                       */
/*                                               */
/*************************************************/

cNupletPtsHomologues::cNupletPtsHomologues(int aNb,double aPds) :
    mPts (aNb),   // De taille aNb avec N element par defaut
    mPds (aPds),
    mFlagDr (0)
{
}

bool cNupletPtsHomologues::IsDr(int aK) const
{
   return IsValideFlagDr(aK) && ( (mFlagDr& (1<<aK)) !=0);
}

void   cNupletPtsHomologues::SetDr(int aK)
{
   AssertIsValideFlagDr(aK);
   mFlagDr |= (1<<aK);
}

bool cNupletPtsHomologues::IsValideFlagDr(int aK) const
{
   return (aK>=0) && (aK<31) && (aK<int(mPts.size()));
}

void cNupletPtsHomologues::AssertIsValideFlagDr(int aK) const
{
   ELISE_ASSERT(IsValideFlagDr(aK),"cNupletPtsHomologues::AssertIsValideFlagDr");
}

void cNupletPtsHomologues::AddPts(const Pt2dr & aPt)
{
   mPts.push_back(aPt);
}


const REAL & cNupletPtsHomologues::Pds() const {return mPds;}
REAL & cNupletPtsHomologues::Pds() {return mPds;}


int cNupletPtsHomologues::NbPts() const
{
   return (int)mPts.size();
}

const Pt2dr & cNupletPtsHomologues::PK(int aK) const
{
   ELISE_ASSERT((aK>=0) && (aK<int(mPts.size())),"cNupletPtsHomologues::PK");
   return mPts[aK];
}
Pt2dr & cNupletPtsHomologues::PK(int aK)
{
   ELISE_ASSERT((aK>=0) && (aK<int(mPts.size())),"cNupletPtsHomologues::PK");
   return mPts[aK];
}


void cNupletPtsHomologues::write(class  ELISE_fp & aFile) const
{
     aFile.write(int(mPts.size()));
     aFile.write(mPds);

     for (int aK=0 ; aK<int(mPts.size()) ; aK++)
        aFile.write(mPts[aK]);

}

cNupletPtsHomologues cNupletPtsHomologues::read(ELISE_fp & aFile)
{
   int aNb = aFile.read((int *) 0);
   REAL  aPds =  aFile.read((REAL *) 0);

   cNupletPtsHomologues aRes(aNb,aPds);

    for (int aK=0 ; aK< aNb ; aK++)
        aRes.mPts[aK] = aFile.read((Pt2dr *) 0);

   return aRes;
}

void cNupletPtsHomologues::AssertD2() const
{
   ELISE_ASSERT(mPts.size()==2,"cNupletPtsHomologues::AssertD2");
}

ElCplePtsHomologues & cNupletPtsHomologues::ToCple()
{
   AssertD2();
   return static_cast<ElCplePtsHomologues &> (*this);
}
const ElCplePtsHomologues & cNupletPtsHomologues::ToCple() const
{
   AssertD2();
   return static_cast<const ElCplePtsHomologues &> (*this);
}

const Pt2dr & cNupletPtsHomologues::P1() const
{
   AssertD2();
   return mPts[0];
}
Pt2dr & cNupletPtsHomologues::P1()
{
   AssertD2();
   return mPts[0];
}


const Pt2dr & cNupletPtsHomologues::P2() const
{
   AssertD2();
   return mPts[1];
}
Pt2dr & cNupletPtsHomologues::P2()
{
   AssertD2();
   return mPts[1];
}

/*************************************************/
/*                                               */
/*    ElCplePtsHomologues                        */
/*                                               */
/*************************************************/

const Pt2dr & ElCplePtsHomologues::P1() const {return PK(0);}
Pt2dr & ElCplePtsHomologues::P1() {return PK(0);}


const Pt2dr & ElCplePtsHomologues::P2() const {return PK(1);}
Pt2dr & ElCplePtsHomologues::P2() {return PK(1);}




void  ElCplePtsHomologues::SelfSwap()
{
   ElSwap(P1(),P2());
}


ElCplePtsHomologues::ElCplePtsHomologues(Pt2dr aP1,Pt2dr aP2,REAL aPds) :
   cNupletPtsHomologues(2,aPds)
{
   P1() = aP1;
   P2() = aP2;
}

/*************************************************/
/*                                               */
/*         cPackNupletsHom                       */
/*                                               */
/*************************************************/
cPackNupletsHom::cPackNupletsHom(int aDim) :
   mDim (aDim)
{
}

INT cPackNupletsHom::size() const
{
  return (INT) mCont.size();
}

void cPackNupletsHom::clear()
{
    mCont.clear();
}

cPackNupletsHom::iterator       cPackNupletsHom::begin() {return mCont.begin();}
cPackNupletsHom::const_iterator cPackNupletsHom::begin() const {return mCont.begin();}
cPackNupletsHom::iterator       cPackNupletsHom::end() {return mCont.end();}
cPackNupletsHom::const_iterator cPackNupletsHom::end() const {return mCont.end();}


cNupletPtsHomologues & cPackNupletsHom::back()
{
   ELISE_ASSERT(!mCont.empty(),"Empty in ElPackHomologue::back");
   return mCont.back();
}
const cNupletPtsHomologues & cPackNupletsHom::back()  const
{
   ELISE_ASSERT(!mCont.empty(),"Empty in ElPackHomologue::back");
   return mCont.back();
}

void cPackNupletsHom::AddNuplet(const cNupletPtsHomologues & aNuple)
{
   ELISE_ASSERT(mDim == aNuple.NbPts(),"cNupletPtsHomologues::AddNuplet");
   mCont.push_back(aNuple);
}

cPackNupletsHom::tIter  cPackNupletsHom::NearestIter(Pt2dr aP,int aK)
{
     REAL dMin = -1;
     ElPackHomologue::tIter aRes = mCont.end();

     for (tCont::iterator itCPl=mCont.begin() ; itCPl!=mCont.end() ; itCPl++)
     {
         REAL aD = euclid(aP,itCPl->PK(aK));
         if ((aRes==mCont.end()) ||   (aD<dMin))
         {
              dMin = aD;
              aRes = itCPl;
         }
     }

     return aRes;
}

const cNupletPtsHomologues * cPackNupletsHom::Nuple_Nearest(Pt2dr aP,int aK) const
{
   tIter anIt = const_cast<cPackNupletsHom *>(this)->NearestIter(aP,aK);

   return  (anIt== mCont.end()) ? 0 : &(*anIt);
}

void cPackNupletsHom::Nuple_RemoveNearest(Pt2dr aP,int aK)
{
    tIter anIter = NearestIter(aP,aK);
    if (anIter != mCont.end())
    {
         mCont.erase(anIter);
    }
}

const ElPackHomologue & cPackNupletsHom::ToPckCple() const
{
   return static_cast<const ElPackHomologue &> (*this);
}

/*************************************************/
/*                                               */
/*         ::                                    */
/*                                               */
/*************************************************/

std::vector<Pt3dr> * StdNuage3DFromFile(const std::string & aName)
{
   if (! IsPostfixed(aName))
   {
        std::cout << "For Name " << aName << "\n";
        ELISE_ASSERT(false,"StdNuage3DFromFile name un post-fixed");
   }

   std::string aPost = StdPostfix(aName);

   if (aPost=="dat")
   {
       std::vector<Pt3dr> *  aRes = new std::vector<Pt3dr> ;

		#if ELISE_windows && !ELISE_MinGW
			ifstream f(aName.c_str(), ios::binary);
			ELISE_DEBUG_ERROR( !f, "StdNuage3DFromFile", "failed to open file [" << aName << "]");

			INT4 nbPoints;
			f.read((char *)&nbPoints, 4);
			ELISE_DEBUG_ERROR(nbPoints < 0, "StdNuage3DFromFile", "invalid nbPoints = " << nbPoints);
			aRes->resize((size_t)nbPoints);

			REAL readPoint[3];
			Pt3dr *itDst = aRes->data();
			while (nbPoints--)
			{
				f.read((char *)readPoint, sizeof(readPoint));
				itDst->x = readPoint[0];
				itDst->y = readPoint[1];
				(*itDst++).z = readPoint[2];
			}
		#else
			ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);
			int aNb = aFile.read((INT4 *)0);
			aRes->reserve(aNb);
			for (int aK=0 ; aK<aNb; aK++)
				aRes->push_back(aFile.read((Pt3dr *)0));
		#endif

       return aRes;
   }
   std::cout << "For Name " << aName << "\n";
   ELISE_ASSERT(false,"unsuported post-fix");
   return NULL;
}

/*************************************************/
/*                                               */
/*    ElPackHomologue                            */
/*                                               */
/*************************************************/

ElPackHomologue::ElPackHomologue() :
   cPackNupletsHom(2),
   mSolveInL1 (true)
{
}

ElCplePtsHomologues & ElPackHomologue::Cple_Back() { return back().ToCple(); }
const ElCplePtsHomologues & ElPackHomologue::Cple_Back()  const { return back().ToCple(); }


void  ElPackHomologue::SelfSwap()
{
   for ( iterator itP=begin(); itP!=end(); itP++)
       itP->ToCple().SelfSwap();
}



void ElPackHomologue::Cple_Add(const ElCplePtsHomologues & aCpl)
{
     AddNuplet(aCpl);
}

ElMatrix<REAL> ElPackHomologue::SolveSys(const  ElMatrix<REAL> & Flin)
{
    ELISE_ASSERT(mSolveInL1,"ElPackHomologue::SolveSys");
    Optim_L1FormLin aOL1(Flin);
    return aOL1.Solve();
}

void ElPackHomologue::ApplyHomographies
     (
           const cElHomographie & aH1,
           const cElHomographie & aH2
     )
{
    for (tCont::iterator itCPl=begin() ; itCPl!=end() ; itCPl++)
    {
         itCPl->P1() = aH1.Direct(itCPl->P1());
         itCPl->P2() = aH2.Direct(itCPl->P2());
     }
}




Polynome2dReal  ElPackHomologue::FitPolynome
                (
             bool aL2,
                     INT aDegre,
                     REAL anAmpl,
                     bool aFitX
                )
{

    Polynome2dReal aPol(aDegre,anAmpl);
    INT aNbMon = aPol.NbMonome();
    // ElMatrix<REAL> Flin(aNbMon+1,(INT)size());

    cGenSysSurResol * aSys = 0;
    if (aL2)
        aSys =  new L2SysSurResol(aNbMon);
    else
        aSys = new SystLinSurResolu(aNbMon,(INT)size());

     aSys->SetPhaseEquation(0);

    Im1D_REAL8  aIMCoeff(aNbMon);
    double * aDC = aIMCoeff.data();

    INT kCpl =0;
    for (tCont::iterator itCPl=begin() ; itCPl!=end() ; itCPl++)
    {
         REAL aPds = itCPl->Pds();
         Pt2dr aP1 = itCPl->P1();
         Pt2dr aP2 = itCPl->P2();

         for (INT kMon =0 ; kMon<aNbMon ; kMon++)
         {
               // Flin(kMon,kCpl) = aPol.KthMonome(kMon)(aP1) * aPds;
           aDC[kMon] = aPol.KthMonome(kMon)(aP1);
         }

         // Flin(aNbMon,kCpl) =  - (aFitX ?  aP2.x : aP2.y) * aPds;
     aSys->GSSR_AddNewEquation(aPds,aDC,(aFitX?aP2.x:aP2.y),0);
         kCpl++;
    }

    // ElMatrix<REAL> aSol = SolveSys(Flin);

    Im1D_REAL8 aSol2 = aSys->GSSR_Solve(0);

    for (INT kMon =0 ; kMon<aNbMon ; kMon++)
    {
      //  std::cout << "FIT POL : " << aSol(0,kMon) << " " << aSol2.data()[kMon] << "\n";
      //   aPol.SetCoeff(kMon,aSol(0,kMon));
        aPol.SetCoeff(kMon,aSol2.data()[kMon]);
    }


    delete aSys;

    return aPol;

}


ElDistortionPolynomiale ElPackHomologue::FitDistPolynomiale
                        (
                             bool aL2,
                             INT aDegre,
                             REAL anAmpl,
                             REAL anEpsInv
                        )
{
     return ElDistortionPolynomiale
            (
                 FitPolynome(aL2,aDegre,anAmpl,true),
                 FitPolynome(aL2,aDegre,anAmpl,false),
                 anEpsInv
            );
}



void ElPackHomologue::PrivDirEpipolaire(Pt2dr & aRes1,Pt2dr & aRes2,INT aSz) const
{
    StatElPackH  aStat(*this);


    Im2D_REAL8 aScore(aSz,aSz);
    Im2D_REAL8 aScoreMoy(aSz,aSz);

    for (INT aK1=0 ; aK1<aSz ; aK1++)
    {
        std::cout << "PrivDirEpipolaire Phase1, remain " << (aSz-aK1) << "\n";
        for (INT aK2=0 ; aK2<aSz ; aK2++)
        {
             REAL alpha1 = (aK1 + 0.5)* 3.14 /aSz;
             REAL alpha2 = (aK2 + 0.5)* 3.14 /aSz;
             Pt2dr  aDir1 = Pt2dr::FromPolar(1,alpha1);
             Pt2dr  aDir2 = Pt2dr::FromPolar(1,alpha2);

              CpleEpipolaireCoord * aCple = CpleEpipolaireCoord::PolynomialFromHomologue
                                            (
                                                  true,
                                                  *this,
                                                  1,
                                                  aDir1,
                                                  aDir2
                                            );
              REAL aBig = 1e3;
              Pt2dr aDirEpi1 = aCple->EPI1().DirEpip(aStat.Cdg1(),aStat.RMax1()/aBig);
              Pt2dr aDirEpi2 = aCple->EPI2().DirEpip(aStat.Cdg2(),aStat.RMax2()/aBig);

              REAL eCart1 = angle_de_droite_nor(aDir1,aDirEpi1);
              REAL eCart2 = angle_de_droite_nor(aDir2,aDirEpi2);


              aScore.data()[aK2][aK1] = eCart1+eCart2;
              delete aCple;
        }
    }
    {
        for (INT aK1=0 ; aK1<aSz ; aK1++)
        {
            std::cout << "PrivDirEpipolaire Phase1, remain " << (aSz-aK1) << "\n";
            for (INT aK2=0 ; aK2<aSz ; aK2++)
            {
                 REAL ecart;
                 ELISE_COPY
                 (
                     aScore.all_pts(),
                     aScore.in()*(1-ecart_frac((aK1-FX)/REAL(aSz))-ecart_frac((aK2-FY)/REAL(aSz))),
                     sigma(ecart)
                 );
                  aScoreMoy.data()[aK2][aK1] = ecart;
            }
        }
    }




    Pt2di aPK;
    ELISE_COPY(aScoreMoy.all_pts(),aScoreMoy.in(),aPK.WhichMin());


    aRes1 = Pt2dr::FromPolar(1,(aPK.x + 0.5)* 3.14 /aSz);
    aRes2 = Pt2dr::FromPolar(1,(aPK.y + 0.5)* 3.14 /aSz);


}

CpleEpipolaireCoord *  ElPackHomologue::DirAndCpleEpipolaire
                       (Pt2dr & aDir1,Pt2dr & aDir2,INT aNbWanted,INT aNbDir,INT aDegreFinal) const
{
    INT Rest = size();
    INT Wanted = aNbWanted;


     ElPackHomologue aSub;

     for (const_iterator anIt=begin() ; anIt != end() ; anIt++)
     {
          if (NRrandom3() *Rest <= Wanted)
          {
               aSub.Cple_Add(anIt->ToCple());
               Wanted--;
          }
          Rest--;
     }

    aSub.PrivDirEpipolaire(aDir1,aDir2,aNbDir);

    CpleEpipolaireCoord * aCple = CpleEpipolaireCoord::PolynomialFromHomologue
                                  (
                                     true,
                                     *this,
                                     aDegreFinal,
                                     aDir1,
                                     aDir2
                                  );
    StatElPackH  aStat(*this);
    REAL aBig = 1e3;
    aDir1 = aCple->EPI1().DirEpip(aStat.Cdg1(),aStat.RMax1()/aBig);
    aDir2 = aCple->EPI2().DirEpip(aStat.Cdg2(),aStat.RMax2()/aBig);


    return  aCple;

}

void  ElPackHomologue::DirEpipolaire
      (Pt2dr & aDir1,Pt2dr & aDir2,INT aNbWanted,INT aNbDir,INT aDegre) const
{
    CpleEpipolaireCoord * aCple = DirAndCpleEpipolaire(aDir1,aDir2,aNbWanted,aNbDir,aDegre);
    delete aCple;
}


void ElPackHomologue::InvY(Pt2dr aSzIm1,Pt2dr aSzIm2)
{
    for (tCont::iterator itCPl=begin() ; itCPl!=end() ; itCPl++)
    {
         itCPl->P1().y = aSzIm1.y-itCPl->P1().y;
         itCPl->P2().y = aSzIm2.y-itCPl->P2().y;
     }
}

void ElPackHomologue::Resize(double aRatioIm1,double aRatioIm2)
{
    for (tCont::iterator itCPl=begin() ; itCPl!=end() ; itCPl++)
    {
         itCPl->P1() = itCPl->P1()*aRatioIm1 ;
         itCPl->P2() = itCPl->P2()*aRatioIm2;
     }
}


const ElCplePtsHomologues * ElPackHomologue::Cple_Nearest(Pt2dr aP,bool P1) const
{
   return static_cast<const ElCplePtsHomologues *> (Nuple_Nearest(aP,P1?0:1));
   // return &(*(const_cast<ElPackHomologue *>(this)->NearestIter(aP,P1)));
}


void   ElPackHomologue::Cple_RemoveNearest(Pt2dr aP,bool P1)
{
       Nuple_RemoveNearest(aP,P1?0:1);
}


StatElPackH::StatElPackH
(
     const  ElPackHomologue & aPackH
) :
    mSPds(0.0),
    mNbPts(0),
    mCdg1(0,0),
    mCdg2(0,0),
    mRMax1(0),
    mRMax2(0),
    mSomD1 (0),
    mSomD2 (0)
{
    for
    (
        ElPackHomologue::const_iterator itC = aPackH.begin();
        itC != aPackH.end();
        itC++
    )
    {
        mNbPts++;
        mSPds += itC->Pds();
        mCdg1 += itC->P1() * itC->Pds();
        mCdg2 += itC->P2() * itC->Pds();
    }

    mCdg1 = mCdg1 / mSPds;
    mCdg2 = mCdg2 / mSPds;

    {
      for
      (
          ElPackHomologue::const_iterator itC = aPackH.begin();
          itC != aPackH.end();
          itC++
      )
      {
          double aD1 = euclid(itC->P1(),mCdg1);
          double aD2 = euclid(itC->P2(),mCdg2);
          ElSetMax(mRMax1,aD1);
          ElSetMax(mRMax2,aD2);
          mSomD1 +=  aD1;
          mSomD2 +=  aD2;
      }
    }
}

Pt2dr StatElPackH::Cdg1 ()  const { return mCdg1; }
Pt2dr StatElPackH::Cdg2 ()  const { return mCdg2; }
REAL  StatElPackH::RMax1 () const { return mRMax1; }
REAL  StatElPackH::RMax2 () const { return mRMax2; }
INT   StatElPackH::NbPts()  const { return mNbPts;}
REAL  StatElPackH::SomD1 () const { return mSomD1;}
REAL  StatElPackH::SomD2 () const { return mSomD2;}


// TO FINISH

void Verif(const double & aV,const std::string & aName)
{
    if (std_isnan(aV))
    {
        std::cout << "in File " << aName << "\n";
        ELISE_ASSERT(false,"Pb in Pt reading");
    }
}

void Verif(const Pt2dr & aP,const std::string & aName)
{
   Verif(aP.x,aName);
   Verif(aP.y,aName);
}
void Verif(const ElPackHomologue & aPack,const std::string & aName)
{
     for (ElPackHomologue::const_iterator itP=aPack.begin();itP!=aPack.end();itP++)
     {
         Verif(itP->P1(),aName);
         Verif(itP->P2(),aName);
     }
}

ElPackHomologue ElPackHomologue::FromFile(const std::string & aName)
{
    if (IsPostfixed(aName))
    {
       if (StdPostfix(aName)=="xml")
       {
           cElXMLTree aTree (aName);
           ElPackHomologue aPck= aTree.GetPackHomologues("ListeCpleHom");
           Verif(aPck,aName);
           return aPck;
       }
       if (StdPostfix(aName)=="tif")
       {
          cElImPackHom aIP(aName);
          ElPackHomologue aPck=  aIP.ToPackH(0);
          Verif(aPck,aName);
          return aPck;
       }

       if (StdPostfix(aName)=="dat")
       {
            ELISE_fp aFP(aName.c_str(),ELISE_fp::READ);
            ElPackHomologue aPck = ElPackHomologue::read(aFP);
            aFP.close(true);
            Verif(aPck,aName);
            return aPck;
       }

       if (StdPostfix(aName)=="txt")
       {
            bool End= false;
            string aBuf; //char aBuf[200]; TEST_OVERFLOW

            ElPackHomologue aPck;
            ELISE_fp aFTxt(aName.c_str(),ELISE_fp::READ);

            while (! End)
            {
                if ( aFTxt.fgets( aBuf, End ) ) //if (aFTxt.fgets(aBuf,200,End)) TEST_OVERFLOW
                {
                   Pt2dr aP1,aP2;
                   double aPds=1.0;
                   int aNb = sscanf(aBuf.c_str(),"%lf %lf %lf %lf %lf",&aP1.x,&aP1.y,&aP2.x,&aP2.y,&aPds);

                   if ((aNb==4) || (aNb==5))
                   {
                        aPck.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));
                   }
               /*
                   if (sscanf(aBuf.c_str(),"%lf %lf %lf %lf",&aP1.x,&aP1.y,&aP2.x,&aP2.y)==4) //sscanf(aBuf.c_str(),"%lf %lf %lf %lf",&aP1.x,&aP1.y,&aP2.x,&aP2.y); TEST_OVERFLOW
                     aPck.Cple_Add(ElCplePtsHomologues(aP1,aP2,1.0));
*/
                }
            }


            aFTxt.close(true);
            Verif(aPck,aName);
            return aPck;
       }

    }

    std::cout << "NAME FILE=" << aName << "\n";
    ELISE_ASSERT(false,"Cannot open file");
    return ElPackHomologue();
}

ElPackHomologue   ElPackHomologue::FiltreByFileMasq
                  (
                         const std::string & aName,
                         double aVMin
                  )  const
{
   ElPackHomologue aRes;
   Im2D_U_INT1 aIm = Im2D_U_INT1::FromFileStd(aName);
   TIm2D<U_INT1,INT> aTIm(aIm);

   for (tCstIter itP=begin(); itP!=end(); itP++)
   {
       double aV = aTIm.get(Pt2di(itP->P1()),0);
       if (aV>=aVMin)
          aRes.Cple_Add(itP->ToCple());
   }
   return aRes;
}

void ElPackHomologue::StdAddInFile(const std::string & aName) const
{
   if (!ELISE_fp::exist_file(aName))
   {
      StdPutInFile(aName);
      return;
   }
   ElPackHomologue aPack = FromFile(aName);
   aPack.Add(*this);
   aPack.StdPutInFile(aName);
}

void  ElPackHomologue::Add(const  ElPackHomologue & aPack)
{
     for (const_iterator itP=aPack.begin();itP!=aPack.end();itP++)
       Cple_Add(itP->ToCple());
}


void ElPackHomologue::StdPutInFile(const std::string & aName) const
{
  std::string aPost = StdPostfix(aName);
  if (aPost=="dat")
  {
         ELISE_fp aFP(aName.c_str(),ELISE_fp::WRITE);
         write(aFP);
         aFP.close();
  }
  else if (aPost=="xml")
  {
         cElXMLFileIn aFileXML(aName);
         aFileXML.PutPackHom(*this);
  }
  else if (aPost=="txt")
  {
       FILE * aFP = FopenNN(aName,"w","ElPackHomologue::StdPutInFile");
       for
       (
           ElPackHomologue::const_iterator itP=begin();
           itP!=end();
           itP++
       )
       {
           fprintf(aFP,"%f %f %f %f %f\n",itP->P1().x,itP->P1().y,itP->P2().x,itP->P2().y,itP->Pds());
       }
       ElFclose(aFP);
/*
      if ( (! OkPt(itP->P1(),aSz1))  || (! OkPt(itP->P2(),aSz2)))
         ELISE_fp aFP(aName.c_str(),ELISE_fp::WRITE);
         write(aFP);
         aFP.close();
*/
  }
  else
  {
     std::cout << "NAME = " << aName << "\n";
     ELISE_ASSERT
     (
        false,
        "Extension de fichier non gere dans ElPackHomologue::StdPutInFile"
     );
  }

}


/*************************************************/
/*                                               */
/*             cElImPackHom                      */
/*                                               */
/*************************************************/

cElImPackHom::cElImPackHom
(
     const ElPackHomologue & aPack,
     int mSsResol,
     Pt2di aSzR
) :
  mSz  (aSzR),
  mImX1(aSzR.x,aSzR.y),
  mImY1(aSzR.x,aSzR.y)
{
   mImXn.push_back(Im2D_REAL4(aSzR.x,aSzR.y));
   mImYn.push_back(Im2D_REAL4(aSzR.x,aSzR.y));
   mImPdsN.push_back(Im2D_REAL4(aSzR.x,aSzR.y,0.0));

   TIm2D<REAL4,REAL8> aTX1(mImX1);
   TIm2D<REAL4,REAL8> aTY1(mImY1);
   TIm2D<REAL4,REAL8> aTX2(mImXn[0]);
   TIm2D<REAL4,REAL8> aTY2(mImYn[0]);
   TIm2D<REAL4,REAL8> aTPds(mImPdsN[0]);


   for (ElPackHomologue::tCstIter itP= aPack.begin() ;  itP!=aPack.end() ; itP++)
   {
        Pt2dr aP1 = itP->P1();
        Pt2dr aP2 = itP->P2();
    // Pt2dr aFrac = aP1-round_ni(aP1);
    Pt2di aPInd (round_down(aP1.x/mSsResol),round_down(aP1.y/mSsResol));
    //ELISE_ASSERT(euclid(aFrac)<1e-4,"Non Int in cElImPackHom::cElImPackHom");
    if (aTPds.get(aPInd)!=0.0)
    {
        std::cout << "PDS  " << aPInd << "=" << aTPds.get(aPInd) << " NEW PDS " << itP->Pds() << "\n";
        std::cout << aP1 << aP2 << "\n";
        std::cout <<  "OLD P1 " << aTX1.get(aPInd) << " " << aTY1.get(aPInd) << "\n";
        std::cout <<  "RESOL  " <<  mSsResol << "\n";

        ELISE_ASSERT(false,"Multiple indexe in cElImPackHom::cElImPackHom");
        }

    aTX1.oset(aPInd,aP1.x);
    aTY1.oset(aPInd,aP1.y);
    aTX2.oset(aPInd,aP2.x);
    aTY2.oset(aPInd,aP2.y);
    aTPds.oset(aPInd,itP->Pds());
   }

}

cElImPackHom::cElImPackHom(const std::string & aNF) :
   mSz (Tiff_Im::BasicConvStd(aNF).sz()),
   mImX1 (mSz.x,mSz.y),
   mImY1 (mSz.x,mSz.y)
{
   mImXn.push_back(Im2D_REAL4(mSz.x,mSz.y));
   mImYn.push_back(Im2D_REAL4(mSz.x,mSz.y));
   mImPdsN.push_back(Im2D_REAL4(mSz.x,mSz.y));

   Tiff_Im aTF = Tiff_Im::BasicConvStd(aNF);
   ELISE_ASSERT
   (
       aTF.phot_interp()==Tiff_Im::PtDeLiaison,
       "Bas Type in cElImPackHom::cElImPackHom"
   );

   ELISE_COPY
   (
       aTF.all_pts(),
       aTF.in(),
       Virgule(mImPdsN[0].out(),mImX1.out(),mImY1.out(),mImXn[0].out(),mImYn[0].out())
   );
}


void cElImPackHom::AddFile(const std::string & aName)
{
    cElImPackHom aPack2(aName);
    ELISE_ASSERT(mSz==aPack2.mSz,"Sz Diff in cElImPackHom::AddFile");

    double aMaxDif;
    ELISE_COPY
    (
        select
    (
        mImX1.all_pts(),
        (mImPdsN[0].in()>1e-5) && (aPack2.mImPdsN[0].in()>1e-5)
        ),
    Abs(mImX1.in()-aPack2.mImX1.in())+ Abs(mImY1.in()-aPack2.mImY1.in()),
        VMax(aMaxDif)
    );

    ELISE_COPY
    (
        select (mImX1.all_pts(), (aPack2.mImPdsN[0].in()>1e-5)),
    Virgule(aPack2.mImX1.in(),aPack2.mImY1.in()),
    Virgule(mImX1.out(),mImY1.out())
    );

    ELISE_ASSERT(aMaxDif<1e-5,"P1 dif in cElImPackHom::AddFile");

    ELISE_ASSERT(aMaxDif<1e-5,"P1 dif in cElImPackHom::AddFile");
    mImXn.push_back(aPack2.mImXn[0]);
    mImYn.push_back(aPack2.mImYn[0]);
    mImPdsN.push_back(aPack2.mImPdsN[0]);
}



ElPackHomologue  cElImPackHom::ToPackH(int aK)
{
   ELISE_ASSERT(aK<NbIm()-1,"Bad Nb Im in cElImPackHom::ToPackH");

   TIm2D<REAL4,REAL8> aTX1(mImX1);
   TIm2D<REAL4,REAL8> aTY1(mImY1);
   TIm2D<REAL4,REAL8> aTX2(mImXn[aK]);
   TIm2D<REAL4,REAL8> aTY2(mImYn[aK]);
   TIm2D<REAL4,REAL8> aTPds(mImPdsN[aK]);

   ElPackHomologue aRes;

   Pt2di aP;
   for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
   {
       for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
       {
          if (aTPds.get(aP))
      {
         aRes.Cple_Add
         (
              ElCplePtsHomologues
          (
               Pt2dr(aTX1.get(aP),aTY1.get(aP)),
               Pt2dr(aTX2.get(aP),aTY2.get(aP)),
               aTPds.get(aP)
          )
         );
          }
       }
   }

   return aRes;
}



int cElImPackHom::NbIm() const{return (int)(1 + mImXn.size());}
void  cElImPackHom::SauvFile(const std::string & aName)
{
    ELISE_ASSERT(NbIm()==2,"Bad Nb Im in cElImPackHom::SauvFile");
    Tiff_Im  aTF
             (
                aName.c_str(),
        mImX1.sz(),
                GenIm::real4,
        Tiff_Im::No_Compr,
        Tiff_Im::PtDeLiaison
         );

    ELISE_COPY
    (
       aTF.all_pts(),
       Virgule(mImPdsN[0].in(),mImX1.in(),mImY1.in(),mImXn[0].in(),mImYn[0].in()),
       aTF.out()
    );
}


Pt2di cElImPackHom::Sz() const
{
   return mSz;
}

void cElImPackHom::VerifInd(Pt2di aP)
{
   ELISE_ASSERT
   (
       (aP.x>=0)&&(aP.y>=0)&&(aP.x<mSz.x)&&(aP.y<mSz.y),
       "cElImPackHom::VerifInd, P"
   );
}

void cElImPackHom::VerifInd(Pt2di aP,int aK)
{
   VerifInd(aP);
   ELISE_ASSERT((aK>=0) && (aK<int(mImXn.size())), "cElImPackHom::VerifInd, K");
}

Pt2dr cElImPackHom::P1(Pt2di aP)
{
   VerifInd(aP);
   return Pt2dr(mImX1.data()[aP.y][aP.x],mImY1.data()[aP.y][aP.x]);
}

Pt2dr cElImPackHom::PN(Pt2di aP,int aK)
{
   VerifInd(aP,aK);
   return Pt2dr(mImXn[aK].data()[aP.y][aP.x],mImYn[aK].data()[aP.y][aP.x]);
}

double cElImPackHom::PdsN(Pt2di aP,int aK)
{
   VerifInd(aP,aK);
   return mImPdsN[aK].data()[aP.y][aP.x];
}



/*************************************************/
/*                                               */
/*    ElProj32                                   */
/*                                               */
/*************************************************/

ElMatrix<REAL> ElProj32::Diff(Pt3dr p) const
{
     ElMatrix<REAL> M(3,2);
     Diff(M,p);
     return M;
}

/*************************************************/
/*                                               */
/*      ProjStenopeGen<Type>                     */
/*                                               */
/*************************************************/


template <class Type> ElProjStenopeGen<Type>::ElProjStenopeGen
                      (
                           Type FOCALE,
                           Type CX,
                           Type CY,
                           const std::vector<Type> & ParamAF
                       )  :
   _focale    (FOCALE),
   _cx        (CX),
   _cy        (CY),
   mUseAFocal ((ParamAF.size() != 0)),
   mParamAF   (ParamAF)
{
   ELISE_ASSERT
   (
           (ParamAF.size()==0) || (ParamAF.size()==NbParamAF)   ,
           "Bad Nb Param afocal in formal ElProjStenopeGen"
   );

}

template <class Type> bool ElProjStenopeGen<Type>::UseAFocal() const
{
   return mUseAFocal;
}

template <class Type> const std::vector<Type>  & ElProjStenopeGen<Type>::ParamAF() const
{
   return mParamAF;
}

template <class Type> Type ElProjStenopeGen<Type>::focale() const
{return _focale;}


template <class Type> Type & ElProjStenopeGen<Type>::focale()
{return _focale;}

template <class Type> Pt2d<Type>  ElProjStenopeGen<Type>::PP()  const
{return Pt2d<Type>(_cx,_cy);}


template <class Type> void ElProjStenopeGen<Type>::SetPP(const  Pt2d<Type> & aPP)
{
  _cx = aPP.x;
  _cy = aPP.y;
}

template <class Type> Type ElProjStenopeGen<Type>::DeltaCProjDirTer
                           (
                             Type x3,
                             Type y3,
                             Type z3
                           ) const
{
      if (! mUseAFocal) return 0;
      Type  r2= Square(x3)+Square(y3);
      Type  R2= r2 + Square(z3);

      Type  F1 =  (1-z3/sqrt(R2)) ; // 1 - cos Teta
      Type  F2 =  Square(F1);

      return    mParamAF[0]*F1 + mParamAF[1]*F2;
}


template <class Type> Type ElProjStenopeGen<Type>::DeltaCProjTer
                           (
                             Type x3,
                             Type y3,
                             Type z3
                           ) const
{
     if (! mUseAFocal) return 0;
         // APPROXIMATION  !!!!!!!
    return DeltaCProjDirTer(x3,y3,z3);

}


template <class Type> void ElProjStenopeGen<Type>::Proj
                       (Type & x2,Type & y2,Type   x3,Type y3,Type z3) const
{
    Type  aZCor = z3;
    if (mUseAFocal)
    {
         aZCor = aZCor - DeltaCProjTer(x3,y3,z3);
    }
    x2 = x3 * (_focale/aZCor) + _cx;
    y2 = y3 * (_focale/aZCor) + _cy;
}

template <class Type> void ElProjStenopeGen<Type>::DirRayon
                       (Type & x3,Type & y3,Type & z3,Type x2,Type y2) const
{
   x3 = (x2-_cx)/_focale;
   y3 = (y2-_cy)/_focale;
   z3 = 1.0;
}

template <class Type> void ElProjStenopeGen<Type>::Diff
                          (ElMatrix<Type> & M,Type x3,Type y3,Type z3) const
{
     ELISE_ASSERT(!mUseAFocal,"ElProjStenopeGen::Diff aFocale");
     M.set_to_size(3,2);

     M(0,0) = _focale/z3;
     M(0,1) = 0;
     M(1,0) = 0;
     M(1,1) = _focale/z3;
     M(2,0) = - (x3 * _focale) / (z3*z3);
     M(2,1) = - (y3 * _focale) / (z3*z3);
}

template <class Type> Type ElProjStenopeGen<Type>::DeltaCProjIm(Type x2,Type y2) const
{
     if (!mUseAFocal)
     {
         return 0.0;
     }
     Type x3,y3,z3;
     DirRayon(x3,y3,z3,x2,y2);
    return DeltaCProjDirTer(x3,y3,z3);
}

template class ElProjStenopeGen<double>;
template class ElProjStenopeGen<Fonc_Num>;



/*************************************************/
/*                                               */
/*    ElProjStenope                              */
/*                                               */
/*************************************************/

ElProjStenope::ElProjStenope(REAL Focale,Pt2dr Centre,const std::vector<double> & aParamAF) :
     ElProjStenopeGen<REAL>(Focale,Centre.x,Centre.y,aParamAF)
{
}

Pt3dr   ElProjStenope::CentreProjIm(const Pt2dr & aP) const
{
   return Pt3dr(0,0,ElProjStenopeGen<REAL>::DeltaCProjIm(aP.x,aP.y));
}

Pt3dr   ElProjStenope::CentreProjTer(const Pt3dr & aP) const
{
   return Pt3dr(0,0,ElProjStenopeGen<REAL>::DeltaCProjTer(aP.x,aP.y,aP.z));
}


Pt2dr ElProjStenope::Proj(Pt3dr p3) const
{
     Pt2dr res;
     ElProjStenopeGen<REAL>::Proj(res.x,res.y,p3.x,p3.y,p3.z);
     return res;
}
Pt3dr ElProjStenope::DirRayon(Pt2dr p2) const
{
   Pt3dr res;
   ElProjStenopeGen<REAL>::DirRayon(res.x,res.y,res.z,p2.x,p2.y);
   return res;
}


void ElProjStenope::Rayon(Pt2dr aP,Pt3dr &p0,Pt3dr & p1) const
{
    // p0 = Pt3dr(0,0,0);
    p0 = CentreProjIm(aP);
    p1 = p0+ DirRayon(aP);
}

Pt2dr ElProjStenope::centre() const
{
    return Pt2dr(_cx,_cy);
}

void  ElProjStenope::set_centre(Pt2dr c)
{
    _cx = c.x;
    _cy = c.y;
}


void ElProjStenope::Diff(ElMatrix<REAL> & M,Pt3dr p) const
{
    ElProjStenopeGen<REAL>::Diff(M,p.x,p.y,p.z);
}

/*

Pt2dr ElProjStenope::Proj(Pt3dr p) const
{
    return Pt2dr(p.x,p.y)* (_focale/p.z) + _centre;
}

Pt3dr ElProjStenope::DirRayon(Pt2dr p) const
{
      return Pt3dr
             (
                (p.x-_centre.x)/_focale,
                (p.y-_centre.y)/_focale,
                1.0
             );
}

Pt2dr   ElProjStenope::centre() const {return _centre;}
Pt2dr & ElProjStenope::centre()       {return _centre;}
*/



/*************************************************/
/*                                               */
/*    cElDistFromCam                             */
/*                                               */
/*************************************************/

cElDistFromCam::cElDistFromCam(const ElCamera & aCam,bool UseRay) :
   mCam     (aCam),
   mUseRay  (UseRay && mCam.HasRayonUtile()),
   mSzC     (mCam.Sz()),
   mMil     (mSzC/2.0),
   mRayU    (mUseRay ? mCam.RayonUtile() : -1)
{
    if (mUseRay)
       mEpsInvDiff = 1e-3;
}

Pt2dr cElDistFromCam::Direct(Pt2dr aP) const
{
   if (! mUseRay)
       return mCam.DistDirecte(aP);

    double aD = euclid(aP,mMil);
    if (aD<mRayU)
       return mCam.DistDirecte(aP);

    double aEps = 1e-3;
    Pt2dr aPC = mMil + vunit(aP-mMil) * mRayU;
    Pt2dr aDPC = mCam.DistDirecte(aPC);
    Pt2dr aDx =( mCam.DistDirecte(aPC+Pt2dr(aEps,0))-aDPC) /aEps;
    Pt2dr aDy =( mCam.DistDirecte(aPC+Pt2dr(0,aEps))-aDPC) /aEps;

    Pt2dr aPP = aP-aPC;

    return aDPC + aDx*aPP.x + aDy*aPP.y;

}

bool cElDistFromCam::OwnInverse(Pt2dr & aP) const
{
   if (mUseRay)
      return false;
   aP = mCam.DistInverse(aP);
   return true;
}
void  cElDistFromCam::Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
{
   DiffByDiffFinies(aMat,aP,euclid(mSzC)/400.0);
}


/*************************************************/
/*                                               */
/*            cCorrRefracAPost                   */
/*                                               */
/*************************************************/


cCorrRefracAPost::cCorrRefracAPost
(
    const cCorrectionRefractionAPosteriori & aCRAP
)  :
   mXML          (new cCorrectionRefractionAPosteriori(aCRAP)),
   mCamEstim     (Cam_Gen_From_File(aCRAP.FileEstimCam(),aCRAP.NameTag().Val(),0)),
   mCoeffRefrac  (aCRAP.CoeffRefrac()),
   mIntegDist    (aCRAP.IntegreDist().Val())
{
}

const cCorrectionRefractionAPosteriori & cCorrRefracAPost::ToXML() const
{
   return *mXML;
}


//
Pt3dr cCorrRefracAPost::CorrectRefrac(const Pt3dr & aP,double aCoef) const
{
    double aXY2 =  square_euclid(Pt2dr(aP.x,aP.y));
    double aZ2 =  ElSquare(aP.z);

    double aT2 = aXY2/aZ2;
    double aRho2 = ElSquare(aCoef);

    double aDenom = (1+aT2 - aRho2 * aT2);
    ELISE_ASSERT(aDenom>0,"cCorrRefracAPost::CorrectRefrac");

    double aAlpha  = sqrt(aRho2/aDenom);

    return Pt3dr(aP.x*aAlpha,aP.y*aAlpha,aP.z);
}


Pt2dr cCorrRefracAPost::CorrM2C(const Pt2dr & aP0) const
{
   Pt3dr aP =  mIntegDist ?
               mCamEstim->C2toDirRayonL3(aP0) :
               mCamEstim->F2toDirRayonL3(aP0);
   return mCamEstim->L3toF2(CorrectRefrac(aP,mCoeffRefrac));
}

Pt2dr cCorrRefracAPost::CorrC2M(const Pt2dr & aP0) const
{
   Pt3dr aP = CorrectRefrac(mCamEstim->F2toDirRayonL3(aP0),1/mCoeffRefrac);

    return mIntegDist  ?
           mCamEstim->L3toC2(aP):
           mCamEstim->L3toF2(aP);
}

/*************************************************/
/*                                               */
/*    cBasicGeomCap3D                            */
/*                                               */
/*************************************************/

Pt2dr    cBasicGeomCap3D::SzPixel() const {return Pt2dr(SzBasicCapt3D());}

Pt2dr  cBasicGeomCap3D::ImRef2Capteur   (const Pt2dr & aP) const {return aP;}
double  cBasicGeomCap3D::ResolImRefFromCapteur() const  {return 1.0;}

double cBasicGeomCap3D::GetVeryRoughInterProf() const
{
   return 1/600.0;
}

std::string cBasicGeomCap3D::Save2XmlStdMMName
     (
           cInterfChantierNameManipulateur * anICNM,
           const std::string & aOriOut,
           const std::string & aNameImClip,
           const ElAffin2D & anOrIntInit2Cur
     ) const
{
    ELISE_ASSERT(false,"CamStenope::Save2XmlStdMMName Not Suported");
    return "";
}




std::string  cBasicGeomCap3D::Save2XmlStdMMName
      (
           cInterfChantierNameManipulateur * anICNM,
           const std::string & aOriOut,
           const std::string & aNameImClip
      ) const
{
    return Save2XmlStdMMName(anICNM,aOriOut,aNameImClip,ElAffin2D::Id());
}

std::string  cBasicGeomCap3D::Save2XmlStdMMName
      (
           cInterfChantierNameManipulateur * anICNM,
           const std::string & aOriOut,
           const std::string & aNameImClip,
           const Pt2dr & aP
      )  const
{
    return Save2XmlStdMMName(anICNM,aOriOut,aNameImClip,ElAffin2D::trans(aP));
}


Pt2dr cBasicGeomCap3D::Mil() const
{
    return Pt2dr(SzBasicCapt3D() ) / 2.0;
}

bool   cBasicGeomCap3D::HasRoughCapteur2Terrain() const
{
    return true;
}

bool cBasicGeomCap3D::IsRPC() const
{
   return false;
}


/*
Pt2dr cBasicGeomCap3D::OrGlbImaM2C(const Pt2dr & aP) const
{
   return aP;
}
*/

void cBasicGeomCap3D::SetScanImaM2C(const tOrIntIma & aOrM2C)
{
   mScanOrImaM2C = aOrM2C;
   mScanOrImaC2M = mScanOrImaM2C.inv();
   ReCalcGlbOrInt();
}
void cBasicGeomCap3D::SetScanImaC2M(const tOrIntIma & aOrC2M)
{
    SetScanImaM2C(aOrC2M.inv());
}
void cBasicGeomCap3D::SetIntrImaM2C(const tOrIntIma & aOrM2C)
{

   mIntrOrImaM2C = aOrM2C;
   mIntrOrImaC2M = mIntrOrImaM2C.inv();
   ReCalcGlbOrInt();
}
void cBasicGeomCap3D::SetIntrImaC2M(const tOrIntIma & aOrC2M)
{
    SetIntrImaM2C(aOrC2M.inv());
}


Pt2dr  cBasicGeomCap3D::OrGlbImaM2C(const Pt2dr & aP) const
{
   return mGlobOrImaM2C(aP);
}
Pt2dr  cBasicGeomCap3D::OrGlbImaC2M(const Pt2dr & aP) const
{
   return mGlobOrImaC2M(aP);
}
Pt2dr  cBasicGeomCap3D::OrIntrImaC2M(const Pt2dr & aP) const
{
   return mIntrOrImaC2M(aP);
}
Pt2dr  cBasicGeomCap3D::OrScanImaM2C(const Pt2dr & aP) const
{
   return mScanOrImaM2C(aP);
}
void cBasicGeomCap3D::ReCalcGlbOrInt()
{
   mGlobOrImaM2C = mIntrOrImaM2C * mScanOrImaM2C;
   mGlobOrImaC2M = mGlobOrImaM2C.inv();

   mScaleAfnt = euclid(mGlobOrImaM2C.IVect(Pt2dr(1,1))) / euclid(Pt2dr(1,1));
}

cBasicGeomCap3D::~cBasicGeomCap3D() 
{
}

Pt3dr cBasicGeomCap3D::OrigineProf () const
{
   Pt3dr aRes =  OpticalCenterOfPixel(Pt2dr(SzBasicCapt3D()) / 2.0);
   return aRes;
}

cBasicGeomCap3D::cBasicGeomCap3D() :
  mScanOrImaC2M  (tOrIntIma::Id()),
  mIntrOrImaC2M  (tOrIntIma::Id()),
  mGlobOrImaC2M  (tOrIntIma::Id()),
  mScanOrImaM2C  (tOrIntIma::Id()),
  mIntrOrImaM2C  (tOrIntIma::Id()),
  mGlobOrImaM2C  (tOrIntIma::Id()),
  mScaleAfnt       (1.0)
{
}

double  cBasicGeomCap3D::GlobResol() const
{
    return ResolSolOfPt(PMoyOfCenter());
}

Pt3dr  cBasicGeomCap3D::PMoyOfCenter() const
{
    return RoughCapteur2Terrain(Mil());
}

double cBasicGeomCap3D::ProfondeurDeChamps(const Pt3dr & aP) const
{
   Pt2dr aPIm = Ter2Capteur(aP);
   Pt3dr aC = OpticalCenterOfPixel(aPIm);

   return scal(DirVisee(),aP-aC);
}

double cBasicGeomCap3D::ResolutionAngulaire() const
{
    Pt2dr aMil = Pt2dr(SzBasicCapt3D()) / 2.0;
    Pt3dr aPTer = RoughCapteur2Terrain(aMil);

    double aR = ResolSolOfPt(aPTer);
    return   aR / euclid(aPTer-OpticalCenterOfPixel(aMil));
}


Pt3dr cBasicGeomCap3D::DirVisee() const
{
   return DirRayonR3(Pt2dr(SzBasicCapt3D()) / 2.0);
}

bool   cBasicGeomCap3D::HasOpticalCenterOfPixel() const
{
   return true;
}

bool  cBasicGeomCap3D::CaptHasDataGeom(const Pt2dr &) const
{
   return true;
}

Pt3dr    cBasicGeomCap3D::OpticalCenterOfPixel(const Pt2dr & aP) const
{
    ELISE_ASSERT(false,"cBasicGeomCap3D::OpticalCenterOfPixel");
    return Pt3dr(0,0,0);
}

void cBasicGeomCap3D::Diff(Pt2dr & aDx,Pt2dr & aDy,Pt2dr & aDz,const Pt2dr & aPIm,const Pt3dr & aTer)
{
    double aStep = ResolSolOfPt(aTer) / 10.0;

    aDx = (Ter2Capteur(Pt3dr(aTer.x+aStep,aTer.y,aTer.z)) - aPIm) / aStep;
    aDy = (Ter2Capteur(Pt3dr(aTer.x,aTer.y+aStep,aTer.z)) - aPIm) / aStep;
    aDz = (Ter2Capteur(Pt3dr(aTer.x,aTer.y,aTer.z+aStep)) - aPIm) / aStep;
}

CamStenope * cBasicGeomCap3D::DownCastCS() { return 0; }


bool  cBasicGeomCap3D::HasPreciseCapteur2Terrain() const
{
    return false;
}

Pt3dr cBasicGeomCap3D::PreciseCapteur2Terrain   (const Pt2dr & aP) const
{
   ELISE_ASSERT(false,"Camera has no \"PreciseCapteur2Terrain\"  functionality");
   return Pt3dr(0,0,0);
}

/*
    Orientation-_MG_0131.CR2.xml              => Camera Stenope Standard
    UnCor-Orientation-_MG_0065.CR2.xml        => Copie de Camera Stenope Standard
    UnCorExtern-RPC.*txt                      => Copie de RPC

    GB-Orientation-_MG_0065.CR2.xml           =>  Generique Bundle


     ???  => Initial RPC
*/

cBasicGeomCap3D * Polynomial_BGC3M2DNewFromFile (const std::string & aName);

double cBasicGeomCap3D::GetAltiSol() const 
{
   ELISE_ASSERT(false,"cBasicGeomCap3D::GetAltiSol");
   return 0.0;
}
Pt2dr cBasicGeomCap3D::GetAltiSolMinMax() const 
{
   ELISE_ASSERT(false,"cBasicGeomCap3D::GetAltiSolMinMax");
   return Pt2dr(0,0);
}
bool cBasicGeomCap3D::AltisSolIsDef() const 
{
    return false;
}
bool cBasicGeomCap3D::AltisSolMinMaxIsDef() const
{
    return false;
}

double  cBasicGeomCap3D::EpipolarEcart(const Pt2dr & aP1,const cBasicGeomCap3D & aCam2,const Pt2dr & aP2,Pt2dr * aSauvDir) const
{
    const cBasicGeomCap3D & aCam1 = *this;

    ElSeg3D aSeg1 = aCam1.Capteur2RayTer(aP1);
    ElSeg3D aSeg2 = aCam2.Capteur2RayTer(aP2);
    ElSeg3D aSeg1Bis = aCam1.Capteur2RayTer(aP1+Pt2dr(0,1));

    Pt3dr aPInter =  aSeg1.PseudoInter(aSeg2) ;
    double aDist = ElMax(aSeg1Bis.DistDoite(aPInter),aSeg1.DistDoite(aPInter));
     
    Pt3dr aPI2 = aPInter + aSeg2.TgNormee() * aDist;

    Pt2dr aQA = aCam1.Ter2Capteur(aPInter);
    Pt2dr aQB = aCam1.Ter2Capteur(aPI2);

    Pt2dr aDirEpi = vunit(aQB-aQA);

    if (aSauvDir) *aSauvDir = aDirEpi;

    Pt2dr aDif = (aP1- aQA) / aDirEpi;
    return aDif.y;
}



void AutoDetermineTypeTIGB(eTypeImporGenBundle & aType,const std::string & aName)
{               
   if (aType != eTIGB_Unknown) return;
   if (IsPostfixed(aName))
   {
       std::string aPost = StdPostfix(aName);

       if ((aPost=="xml") || (aPost=="XML"))
       {
           std::string aPost2 = aName.substr(aName.size()-7,3);// StdPostfix(aName.substr(0,aName.size()-4));

           //if( aPost2 != "txt" && aPost2 != "TXT" && aPost2 != "Txt" && aPost2 != "rpc")
           if( aPost2 != "txt" && aPost2 != "TXT" && aPost2 != "Txt" )
           {
                cElXMLTree * aTree = new cElXMLTree(aName);

                cElXMLTree * aXmlMETADATA_FORMAT = aTree->Get("METADATA_FORMAT");
                if (aXmlMETADATA_FORMAT)
                {
                    std::string aStrMETADATA_FORMAT = aXmlMETADATA_FORMAT->GetUniqueVal() ;
                    if (aStrMETADATA_FORMAT == "DIMAP")
                    {
                        std::string aStrVersion = aXmlMETADATA_FORMAT->ValAttr("version","-1");
                        if ((aStrVersion =="2.0") || (aStrVersion =="2.12") || (aStrVersion =="2.15"))
                        {
                             //std::cout << "GOT DIMAP2 \n"; getchar();
                            aType = eTIGB_MMDimap2;
                            return;
                        }
                        else if (aStrVersion == "3.0")
                        {
                            //std::cout << "GOT DIMAP3 \n"; getchar();
                            aType = eTIGB_MMDimap3;
                            return;
                        }
                        else
                        {
                            cElXMLTree * aXmlMETADATA_VERSION = aTree->Get("METADATA_VERSION");
                            if(aXmlMETADATA_VERSION)
                            {
                                std::string aStrMETADATA_VERSION = aXmlMETADATA_VERSION->GetUniqueVal() ;
                                if(aStrMETADATA_VERSION == "2.0")
                                {
                                    aType = eTIGB_MMDimap2;
                                    return;
                                }
                            }
                            else
                            {
                                ELISE_ASSERT(false,"AutoDetermineTypeTIGB; A new DIMAP version? We only know versions 2.0, 2.12 and 2.15. Contact developpers for help."); 
                                
                            }
                            
                        }
                    }
                }

                if (     (aTree->Get("NUMROWS") !=0)
                    &&  (aTree->Get("NUMCOLUMNS") !=0)
                    &&  (aTree->Get("ERRBIAS") !=0)
                    &&  (aTree->Get("LINEOFFSET") !=0)
                    &&  (aTree->Get("SAMPOFFSET") !=0)
                    &&  (aTree->Get("LATOFFSET") !=0)
                    &&  (aTree->Get("LONGOFFSET") !=0)
                    &&  (aTree->Get("HEIGHTOFFSET") !=0)
                    &&  (aTree->Get("LINESCALE") !=0)
                    &&  (aTree->Get("SAMPSCALE") !=0)
                    &&  (aTree->Get("LATSCALE") !=0)
                    &&  (aTree->Get("LONGSCALE") !=0)
                    &&  (aTree->Get("HEIGHTSCALE") !=0)
                    &&  (aTree->Get("LINENUMCOEF") !=0)
                    &&  (aTree->Get("LINEDENCOEF") !=0)
                    )
                {
                    aType = eTIGB_MMDGlobe;
                    return;
                }
                
                //Xml_ScanLineSensor
                if (aTree->Get("Xml_ScanLineSensor") !=0)
                {
                    aType = eTIGB_MMScanLineSensor;
                    return;
                }
				//MMEpip
				if (aTree->Get("ListeAppuis1Im") !=0)
				{
					aType = eTIGB_MMEpip;
					return;
				}

           }
           else
           {
                std::string aLine;
                std::ifstream aFile(aName.c_str());
            
                std::getline(aFile, aLine);
                std::getline(aFile, aLine);
                if( aLine.find("<Xml_RPC>") != string::npos )//verify if it's not Xml_RPC
                    aType = eTIGB_MMDimap2;
                else if(aLine.find("TYPE_OBJET") != string::npos)
                    aType = eTIGB_MMEuclid;
                else
                    aType = eTIGB_MMIkonos;
                

           }
       }

       if ((aPost=="txt") || (aPost=="TXT") || (aPost=="rpc"))
       {
            
            std::string aLine;
            std::ifstream aFile(aName.c_str());
            
            std::getline(aFile, aLine);
            if( aLine.find("DATE_GENERATION") != string::npos )
                aType = eTIGB_MMEuclid;
            else
                aType = eTIGB_MMIkonos;
            
            return;
       }

       if ((aPost=="gri") || (aPost=="GRI"))
       {
            aType = eTIGB_MMOriGrille;
            return;
       }
   }
}



cBasicGeomCap3D * cBasicGeomCap3D::StdGetFromFile(const std::string & aName,int & aIntType, const cSystemeCoord * aChSys)
{
    ELISE_ASSERT((aIntType>=0) && (aIntType<eTIGB_NbVals),"cBasicGeomCap3D::StdGetFromFile, Not an  eTypeImporGenBundle");


    eTypeImporGenBundle aType = (eTypeImporGenBundle) aIntType;
    #ifdef REG_EMPTY
        static cElRegex  ThePattMMCS(".*Ori-.*/(UnCorMM-Orientation|Orientation).*xml",10); // MacOS X does not accept empty (sub)expresions
    #else
        static cElRegex  ThePattMMCS(".*Ori-.*/(UnCorMM-|)Orientation.*xml",10);  // Its a stenope Camera created using MicMac
    #endif
    static cElRegex  ThePattGBMM(".*Ori-.*/GB-Orientation-.*xml",10);  // Its a Generik Bundle Camera created using MicMac

    static cElRegex  ThePattSatelit(".*Ori-.*/UnCorExtern-Orientation-(eTIGB_[a-z,A-Z,0-9]*)-.*xml",10);  // Its a copy for generik

   
    if ((aType==eTIGB_MMSten) || ((aType==eTIGB_Unknown) && ThePattMMCS.Match(aName)))
    {
        cElXMLTree aTreeBase (aName);
        // cElXMLTree *  aTreeOri = aTreeBase.GetOneOrZero("OrientationConique");

        if (aTreeBase.GetOneOrZero("OrientationConique"))
        {
             if (aType==eTIGB_Unknown)  aIntType = eTIGB_MMSten;
             return BasicCamOrientGenFromFile(aName);
        }
    }
    else if (aType==eTIGB_MMDimap3 || 
             aType==eTIGB_MMDimap2 || 
             aType==eTIGB_MMDGlobe || 
             aType==eTIGB_MMEuclid || 
             aType==eTIGB_MMIkonos || 
             aType==eTIGB_MMOriGrille ||
             aType==eTIGB_MMScanLineSensor ||
		     aType==eTIGB_MMEpip	)
    {
	
	return CameraRPC::CamRPCOrientGenFromFile(aName, aType, aChSys);
    }

    if (ThePattGBMM.Match(aName))
    {
        return Polynomial_BGC3M2DNewFromFile(aName);
    }

    if (ThePattSatelit.Match(aName))
    {

         std::string aNameType = ThePattSatelit.KIemeExprPar(1);
    
	 eTypeImporGenBundle aTrueType = eTIGB_Unknown;
	 AutoDetermineTypeTIGB(aTrueType,aName);//ER modif to look inside the file rather than reason from the filename: eTypeImporGenBundle aTrueType = Str2eTypeImporGenBundle(aNameType);

         aIntType =  aTrueType;

         switch (aTrueType)
         {
                case eTIGB_MMDGlobe : 
                case eTIGB_MMDimap3 :
                case eTIGB_MMDimap2 :
                case eTIGB_MMEuclid :
                case eTIGB_MMIkonos :
                case eTIGB_MMOriGrille :
                case eTIGB_MMScanLineSensor :
				case eTIGB_MMEpip :
                      return  CameraRPC::CamRPCOrientGenFromFile(aName,aTrueType,aChSys);

                default : ;

         }
           
    }

    std::cout << "For orientation file=" << aName << "\n";
    ELISE_ASSERT(false,"cBasicGeomCap3D::StdGetFromFile"); 

    return 0;
}

cBasicGeomCap3D * cInterfChantierNameManipulateur::StdCamGenerikOfNames(const std::string & anOri,const std::string & aName,bool SVP)
{
   std::string aRes = StdNameCamGenOfNames(anOri,aName);
   int aType = eTIGB_Unknown;


   if (aRes!= "") return cBasicGeomCap3D::StdGetFromFile(aRes,aType);

   if (! SVP)
   {
       std::cout << "For Ori=" << anOri << " , and Name=" << aName << "\n";
       ELISE_ASSERT(false,"cannot get cInterfChantierNameManipulateur::StdCamGenOfNames");
   }

   return 0;

}


std::string  cInterfChantierNameManipulateur::StdNameCamGenOfNames(const std::string & anOri,const std::string & aName)
{
    std::string aN1 = Dir() + Assoc1To1("NKS-Assoc-Im2Orient@-"+anOri,aName,true);
    if (ELISE_fp::exist_file(aN1)) return  aN1;

    std::string aN2 = Dir() + Assoc1To1("NKS-Assoc-Im2GBOrient@-"+anOri,aName,true);
    if (ELISE_fp::exist_file(aN2)) return  aN2;


    return "";
}

/*


cBasicGeomCap3D * StdGetFromFile(const std::string & aName)
{
    
    if (IsPostfixed(aName) &&  (StdPostfix(aName)  == "xml"))
    {
        
    }
}





*/
Pt3dr  cBasicGeomCap3D::ImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
    Pt3dr aC = OpticalCenterOfPixel(aP);
    ElSeg3D aSeg=Capteur2RayTer(aP);

    return aC + aSeg.TgNormee() * aZ;
}

Pt3dr  cBasicGeomCap3D::ImEtZ2Terrain(const Pt2dr & aPIm,double aZ) const
{
    ElSeg3D aSeg = Capteur2RayTer(aPIm);

    Pt3dr aP0 = aSeg.P0();
    Pt3dr aRay =  aSeg.Tgt();
    double aLamda =  (aZ-aP0.z)/aRay.z;

    return aP0 +  aRay * aLamda;
}
Pt3dr cBasicGeomCap3D::DirRayonR3(const Pt2dr & aPIm) const
{
    return Capteur2RayTer(aPIm).TgNormee();
}


void cBasicGeomCap3D::GetCenterAndPTerOnBundle(Pt3dr & aC,Pt3dr & aPTer,const Pt2dr & aPIm) const
{
   ElSeg3D aSeg = Capteur2RayTer(aPIm);

   aC  = OpticalCenterOfPixel(aPIm);
   aPTer = RoughCapteur2Terrain(aPIm);

/*std::cout << aC << "\n";
std::cout << aPTer << "\n";
*/
   aC = aSeg.ProjOrtho(aC);
   aPTer = aSeg.ProjOrtho(aPTer);
/*std::cout << " " <<  aC << "\n";
std::cout << " " << aPTer << "\n";*/
}


/*************************************************/
/*                                               */
/*    ElCamera                                   */
/*                                               */
/*************************************************/

ElCamera::ElCamera(bool isDistC2M,eTypeProj aTP) :

    mTrN       (0,0),
    mScN       (1.0),
    mCRAP      (0),
    _orient (Pt3dr(0,0,0),0,0,0),
    mSz        (-1,-1),
    mSzPixel   (-1,-1),
    mDIsDirect (! isDistC2M),
    mTypeProj  (aTP),
    // mAltisSolIsDef (false),
    // mAltiSol       (-1e30),
    mProfondeurIsDef (false),
    mProfondeur       (0),
    mIdentCam         ("NoCamName"),
    //mPrecisionEmpriseSol (1e30),
    mRayonUtile (-1),
    mHasDomaineSpecial  (false),
    mDoneScanContU      (false),
    mParamGridIsInit (false),
    mTime            (TIME_UNDEF()),
    mScanned         (false),
    mVitesse         (0,0,0),
    mVitesseIsInit   (false),
    mIncCentre       (1,1,1),
    mStatDPCDone     (false)
{
    UndefAltisSol();
}

double ElCamera::GetVeryRoughInterProf() const
{
   return 1/10.0;
}

bool  ElCamera::GetZoneUtilInPixel() const
{
    return ! mScanned;
}



Pt3dr ElCamera::Vitesse() const
{
    ELISE_ASSERT(mVitesseIsInit,"ElCamera::Vitesse nor init");
    return mVitesse;
}

void  ElCamera::SetVitesse(const Pt3dr & aV)
{
    mVitesseIsInit  = true;
    mVitesse = aV;
}

bool  ElCamera::VitesseIsInit() const
{
   return mVitesseIsInit;
}

Pt3dr ElCamera::IncCentre() const
{
    return mIncCentre;
}

void  ElCamera::SetIncCentre(const Pt3dr & anInc)
{
   mIncCentre = anInc;
}

cArgOptionalPIsVisibleInImage::cArgOptionalPIsVisibleInImage() :
    mOkBehind (false)
{
}


bool    ElCamera::PIsVisibleInImage   (const Pt3dr & aPTer,cArgOptionalPIsVisibleInImage * anArg) const
{
   if (anArg) anArg->mWhy ="";

   Pt3dr aPCam = R3toL3(aPTer);


   if (HasOrigineProf())
   {
        double aSeuil = 1e-5 * (ElAbs(aPCam.x)+ElAbs(aPCam.y));
        if (aPCam.z <= aSeuil)
        {
             if ( (anArg==0) || (! anArg->mOkBehind)  || (aPCam.z>-aSeuil))
             {
                if (anArg) anArg->mWhy = "Behind";
                return false;
             }
        }
   }



   Pt2dr aPI0 = Proj().Proj(aPCam);


   if (GetZoneUtilInPixel() )
   {
       Pt2dr aSz = SzPixel();

       Pt2dr aPQ = NormM2C(aPI0) ;
       ElDistortion22_Gen * aDPC = StaticDistPreCond();
       if (aDPC)
       {
/*
                     << " " <<  euclid(aPQ,aDPC->Direct(aPQ)) 
                     << " " <<  euclid(Dist().Direct(aPQ),aDPC->Direct(aPQ)) 
                     << "\n";
*/
           
           aPQ = aDPC->Direct(aPQ);
       }
       double aRab = 0.8;
       Pt2dr aMil = Pt2dr(aSz)/2.0;
   
       aPQ =  aMil+ (aPQ-aMil) * aRab;
       if ((aPQ.x <0)  || (aPQ.y<0) || (aPQ.x>aSz.x) || (aPQ.y>aSz.y))
       {
            if (anArg) anArg->mWhy ="PreCondOut";
            return false;
       }
    }


   Pt2dr aPF0 = DistDirecteSsComplem(aPI0);



   // Si "vraie" camera et scannee il est necessaire de faire le test maintenant
   // car IsZoneUtil est en mm

   if ( (!GetZoneUtilInPixel()) && ( ! IsInZoneUtile(aPF0)))
   {
       if (anArg) anArg->mWhy ="ScanedOut";
       return false;
   }


   Pt2dr aPF1 = DComplM2C(aPF0);

   // MPD le 17/06/2014 : je ne comprend plus le [1], qui fait planter les camera ortho
   // a priori la zone utile se juge a la fin
   if (GetZoneUtilInPixel() && ( ! IsInZoneUtile(aPF1,true))) 
   {
       if (anArg) anArg->mWhy ="NotInImage";
       return false;
   }


   Pt2dr aI0Again = DistInverse(aPF1);


   bool aResult = (euclid(aPI0-aI0Again) < 1.0/ mScaleAfnt);

   if (! aResult)
   {
       if (anArg) anArg->mWhy ="DistCheck";
       return false;
   }

   return aResult;
}


void ElCamera::UndefAltisSol()
{
     mAltisSolIsDef = false;
     mAltiSol       = -1e30;
}

double ElCamera::ResolSolOfPt(const Pt3dr & aP) const
{
    return ResolutionSol(aP);
}

double  ElCamera::ResolSolGlob() const
{
    return ResolutionSol();
}

Pt2di    ElCamera::SzBasicCapt3D() const
{
    return Sz();
}



bool  ElCamera::CaptHasData(const Pt2dr & aP) const
{
   return  IsInZoneUtile(DComplC2M(aP,false));
   // return  IsInZoneUtile(DComplC2M(aP));
   // return  IsInZoneUtile(aP);
}

const bool &   ElCamera::IsScanned() const
{
    return mScanned;
}

void ElCamera::SetScanned(bool mIsSC )
{
    mScanned = mIsSC;
}


Pt2dr    ElCamera::Ter2Capteur   (const Pt3dr & aP) const
{
   return R3toF2(aP);
}
ElSeg3D  ElCamera::Capteur2RayTer(const Pt2dr & aP) const
{
   return F2toRayonR3(aP);
}


Pt2dr ElCamera::ImRef2Capteur   (const Pt2dr & aP) const
{
   return aP;
}

double ElCamera::ResolImRefFromCapteur() const {return 1.0;}

bool  ElCamera::HasRoughCapteur2Terrain() const
{
    return ProfIsDef() || AltisSolIsDef();
}
bool  ElCamera::HasPreciseCapteur2Terrain() const
{
    return false;
}
Pt3dr  ElCamera::RoughCapteur2Terrain   (const Pt2dr & aP) const
{
   if (ProfIsDef())
      return ImEtProf2Terrain(aP,GetProfondeur());

   if (AltisSolIsDef())
      return ImEtZ2Terrain(aP,GetAltiSol());

   ELISE_ASSERT(false,"Nor Alti, nor prof : Camera has no \"RoughCapteur2Terrain\"  functionality");
   return Pt3dr(0,0,0);
}
Pt3dr ElCamera::PreciseCapteur2Terrain   (const Pt2dr & aP) const
{
   ELISE_ASSERT(false,"Camera has no \"PreciseCapteur2Terrain\"  functionality");
   return Pt3dr(0,0,0);
}





const ElAffin2D &  ElCamera::IntrOrImaC2M() const
{
   return mIntrOrImaC2M;
}


const double & ElCamera::GetTime() const
{
   return mTime;
}

void   ElCamera::SetTime(const double & aTime)
{
   mTime = aTime;
}


Pt3dr ElCamera::OrigineProf() const
{
  ELISE_ASSERT(false,"");
  return Pt3dr(0,0,0);
}


bool  ElCamera::HasOrigineProf() const
{
   return false;
}

Pt2dr ElCamera::TrCamNorm() const
{
   return mTrN;
}

double ElCamera::ScaleCamNorm() const
{
   return mScN;
}

void ElCamera::SetParamGrid(const NS_ParamChantierPhotogram::cParamForGrid & aParam)
{
   mParamGridIsInit = true;
   mStepGrid = aParam.StepGrid();
   mRayonInvGrid = aParam.RayonInv();
}

bool ElCamera::IsInZoneUtile(const Pt2dr & aQ,bool Pixel) const
{
   
   // Pt2dr aP = mZoneUtilInPixel ? DComplM2C(aQ) : aQ;
   Pt2dr aP = aQ;
   Pt2di aSz = Pixel ?  Pt2di(SzPixel()) : Sz() ;
   if ((aP.x<=0)  || (aP.y<=0) || (aP.x>=aSz.x) || (aP.y>=aSz.y))
      return false;

   double aR = mRayonUtile;
   if (aR <= 0) return true;
   if (Pixel) aR *= mScaleAfnt;


   return euclid(aP-aSz/2.0) < aR;
}


bool ElCamera::HasRayonUtile() const
{
   return mRayonUtile > 0;
}

double ElCamera::RayonUtile() const
{
   ELISE_ASSERT(HasRayonUtile(),"ElCamera::RayonUtile");
   return mRayonUtile;
}


bool ElCamera::IsForteDist() const
{
   return DistPreCond() != 0;
}

ElDistortion22_Gen   *  ElCamera::StaticDistPreCond() const
{
    if (!mStatDPCDone)
    {
        mStatDPCDone  = true;
        mStatDPC = DistPreCond();
    }
    return mStatDPC;
}

ElDistortion22_Gen   *  ElCamera::DistPreCond() const
{
    return 0;
}

bool ElCamera::IsGrid() const
{
  return false;
}

const Pt2di ElCamera::TheSzUndef(-1,-1);

double  ElCamera::RatioInterSol(const ElCamera & aCam2) const
{
   if (InterVide(mBoxSol,aCam2.mBoxSol))
      return 0.0;

   cElPolygone aPI = mEmpriseSol * aCam2.mEmpriseSol;

   return aPI.Surf() / mEmpriseSol.Surf();
    //mEmpriseSol
}


void  ElCamera::SetProfondeur(double aP)
{
    mProfondeurIsDef = true;
    mProfondeur      = aP;
}

bool ElCamera::ProfIsDef() const { return mProfondeurIsDef; }
bool ElCamera::AltisSolIsDef() const { return mAltisSolIsDef; }

double ElCamera::GetProfondeur() const
{
    ELISE_ASSERT(ProfIsDef(),"Profondeur non init");
    return mProfondeur;
}
double  ElCamera::GetRoughProfondeur() const
{
   return GetProfondeur();
}



const cElPolygone &  ElCamera::EmpriseSol() const
{
    AssertSolInit();
   return mEmpriseSol;
}

const Box2dr &  ElCamera::BoxSol() const
{
    AssertSolInit();
    return mBoxSol;
}





void  ElCamera::SetAltiSol(double  aZ)
{
    mAltisSolIsDef = true;
    mAltiSol       = aZ;

    // Box2dr aBox(Pt2dr(0,0),Pt2dr(Sz()));
    // Pt2dr aP4Im[4];
    // aBox.Corners(aP4Im);


    Pt2dr aP0,aP1;

    std::vector<Pt2dr>  aCont;
    for (int aK=0 ; aK<int(ContourUtile().size()) ; aK++)
    {
         // Pt2dr aCk= OrGlbImaM2C(ContourUtile()[aK]);
         Pt2dr aCk= ContourUtile()[aK];


         Pt3dr aPTer = F2AndZtoR3(aCk,aZ);
         Pt2dr aP2T(aPTer.x,aPTer.y);
         if (aK==0)
         {
            aP0 = aP2T;
            aP1 = aP2T;
         }
         else
         {
            aP0.SetInf(aP2T);
            aP1.SetSup(aP2T);
         }
         aCont.push_back(aP2T);
    }


    mBoxSol = Box2dr(aP0,aP1);
    mEmpriseSol = cElPolygone();
    mEmpriseSol.AddContour(aCont,false);

}

eTypeProj  ElCamera::GetTypeProj() const
{
   return mTypeProj;
}

void ElCamera::AssertSolInit() const
{
   ELISE_ASSERT(mAltisSolIsDef,"ElCamera::GetAltiSol");
}

double ElCamera::GetAltiSol() const
{
   AssertSolInit();
   return mAltiSol;
}

cCamStenopeBilin *  ElCamera::CSBil_SVP()
{
   return 0;
}
cCamStenopeBilin * ElCamera::CSBil()
{
   cCamStenopeBilin * aCSBil = CSBil_SVP();
   ELISE_ASSERT(aCSBil!=0,"ElCamera::CSBil");

   return aCSBil;
}

CamStenope *  ElCamera::CS()
{
    ELISE_ASSERT(mTypeProj==eProjectionStenope,"ElCamera::CS");
    return static_cast<CamStenope * > (this);
}

const CamStenope *  ElCamera::CS() const
{
    ELISE_ASSERT(mTypeProj==eProjectionStenope,"ElCamera::CS");
    return static_cast<const CamStenope * > (this);
}





ElCamera::tOrIntIma  ElCamera::InhibeScaneOri()
{

   tOrIntIma aRes = mScanOrImaM2C;
   mScanOrImaM2C =tOrIntIma::Id();
   ReCalcGlbOrInt();
   return aRes;
}

void ElCamera::RestoreScaneOri(const ElCamera::tOrIntIma & aRestore)
{
   mScanOrImaM2C = aRestore;
   ReCalcGlbOrInt();
}


double  ElCamera::ScaleAfnt() const
{
   return mScaleAfnt;
}

Pt2dr ElCamera::ResiduMond2Cam(const Pt2dr & aRes)const
{
   return mGlobOrImaM2C.IVect(aRes) ;
}








void ElCamera::SetSz(const Pt2di &aSz,bool AcceptInitMult)
{
/*
static int aCpt = 0; aCpt++;
std::cout << "Organge " << aCpt << "\n";
if (aCpt>=8) getchar();
*/

   if ((mSz.x != -1)  && (!AcceptInitMult))
   {
       ELISE_ASSERT(aSz==mSz,"Multiple Sz incoherent in ElCamera::SetSz");
   }
   mSz = aSz;

   Box2dr aBox(Pt2dr(0,0),Pt2dr(Sz()));
   Pt2dr aP4Im[4];
   aBox.Corners(aP4Im);
   if (mContourUtile.empty())
   {
      for (int aK=0 ; aK<4 ; aK++)
      {
         mContourUtile.push_back(aP4Im[aK]);
      }
   }
}

bool ElCamera::HasDomaineSpecial() const
{
   return mHasDomaineSpecial;
}

void   ElCamera::SetRayonUtile(double aRay,int aNbDisc)
{
   mHasDomaineSpecial = true;
   mContourUtile.clear();
   Box2dr aBox(Pt2dr(0,0),Pt2dr(Sz()));
   Pt2dr aMil = Pt2dr(Sz())  / 2.0;
   //double aDiag = euclid(Sz());

//Video_Win *aW = Video_Win::PtrWStd(Pt2di(1000,800));
//double aZ=6;

   for (int aK=0 ; aK<aNbDisc ; aK++)
   {
      Seg2d aSeg(aMil,aMil+Pt2dr::FromPolar(aRay,(2*PI*aK)/aNbDisc));
      Seg2d aSC = aSeg.clip(aBox);
//aW->draw_seg(aSC.p0()/aZ,aSC.p1()/aZ,aW->pdisc()(P8COL::red));
//std::cout << aSC.p0() << aSC.p1() << euclid(aSC.p0(),aSC.p1())<< "\n";
      mContourUtile.push_back(aSC.p1());
   }
//getchar();
   mRayonUtile = aRay;
}

const std::vector<Pt2dr> &  ElCamera::ContourUtile()
{
    ELISE_ASSERT(!mContourUtile.empty(),"ElCamera::ContourUtile non init");
    if (!mDoneScanContU)
    {
       mDoneScanContU = true;
       for (int aK=0 ; aK<int(mContourUtile.size()) ; aK++)
           mContourUtile[aK] = OrScanImaM2C(mContourUtile[aK]);
    }
    return mContourUtile;
}

const Pt2di & ElCamera::Sz() const
{
   ELISE_ASSERT((mSz.x>0),"ElCamera::Sz non initialisee");
   return mSz;
}

bool ElCamera::SzIsInit() const
{
   return mSz.x>0;
}


Box2dr ElCamera::BoxUtile() const
{
    Box2dr aBox(Pt2dr(0,0),Pt2dr(Sz()));
    // return aBox.BoxImage(mGlobOrImaM2C);  ne trouve plus de coherence a cela
    return aBox;
}

Pt2dr ElCamera::DistDirecteSsComplem(Pt2dr aP) const
{
    return mDIsDirect ? Dist().Direct(aP) : Dist().Inverse(aP);
}
Pt2dr ElCamera::DistDirecte(Pt2dr aP) const
{
   // return DComplM2C(mDIsDirect ? Dist().Direct(aP) : Dist().Inverse(aP));
   return DComplM2C(DistDirecteSsComplem(aP));
}


Pt2dr ElCamera::DistInverseSsComplem(Pt2dr aP) const
{
    return  mDIsDirect ? Dist().Inverse(aP) : Dist().Direct(aP);
}


Pt2dr ElCamera::DistInverse(Pt2dr aP) const
{
    return DistInverseSsComplem(DComplC2M(aP));

/*
   aP = DComplC2M(aP);
   aP= mDIsDirect ? Dist().Inverse(aP) : Dist().Direct(aP);
   return aP;
*/
/*
   static int aCpt=0; aCpt++;
   // std::cout << "aCPpppt " <<  aCpt << "\n";
   bool Bug =    (aCpt==149927) || ((aCpt>=159927) && (aCpt<=159930));  //  Avec 1500
   // bool Bug = (aCpt==242400) || ((aCpt>=252400) && (aCpt<=252403));  //  Avec -1
   if (Bug)
   {
       NS_ParamChantierPhotogram::cOrientationConique  aCO = StdExportCalibGlob();
       MakeFileXML(aCO,"Debug-"+ToString(aCpt) + ".xml");
       std::cout << "EXPPPoort ElCamera::DistInverse\n";
   }
*/

}

// void ElCamera::SetDistInverse() { mDIsDirect=false; }
// void ElCamera::SetDistDirecte() { mDIsDirect=true; }


bool ElCamera::DistIsDirecte() const {return mDIsDirect;}
bool ElCamera::DistIsC2M() const {return ! mDIsDirect;}


void  ElCamera::AddDistCompl(bool isDirecte,ElDistortion22_Gen * aDist)
{
   // ELISE_ASSERT(!mDIsDirect ,"ElCamera:: DistDirecte/mDistCompl");
   mDistCompl.push_back(aDist);
   mDComplIsDirect.push_back(isDirecte);
}

void  ElCamera::AddCorrecRefrac(cCorrRefracAPost * aCRAP)
{
    ELISE_ASSERT(mCRAP==0,"Multiple ElCamera::AddCorrecRefrac");
    mCRAP = aCRAP;
}

void ElCamera::TestCam( const std::string & aMes)
{
    std::cout << "TC " << aMes << " "  << this << " "  << mTrN  << mGlobOrImaC2M(Pt2dr(0,0)) << "\n";
}

const std::string &  ElCamera::NameIm() const
{
   return mNameIm;
}
void ElCamera::SetNameIm(const std::string & aName)
{
    mNameIm = aName;
}


const std::string &  ElCamera::IdentCam() const
{
   return mIdentCam;
}
void ElCamera::SetIdentCam(const std::string & aName)
{
   mIdentCam  = aName;
}



void  ElCamera::HeritComplAndSz(const ElCamera & aCam)
{
    CamHeritGen(aCam,true);
}

void  ElCamera::CamHeritGen(const ElCamera & aCam,bool WithCompl,bool WithOrientInterne)
{
   SetIdentCam(aCam.IdentCam());
   SetNameIm(aCam.NameIm());
 //  std::cout << "HHHHHH ii " << this << "\n"; dd

   if (WithOrientInterne)
   {
      SetScanImaM2C(aCam.mScanOrImaM2C);
      SetIntrImaM2C(aCam.mIntrOrImaM2C);
   }


   if (WithCompl)
   {
     mDistCompl = aCam.mDistCompl;
     mDComplIsDirect = aCam.mDComplIsDirect;
     mCRAP = aCam.mCRAP;
     Dist().ScN() = aCam.Dist().ScN();
   }
    mTrN      = aCam.mTrN;
    mScN      = aCam.mScN;

    if (aCam.SzIsInit() && (!SzIsInit()))
    {
       SetSz( aCam.Sz());
    }
    if (aCam.HasRayonUtile() && (!HasRayonUtile()))
    {
      SetRayonUtile(aCam.RayonUtile(),30);
    }
    if (aCam.mParamGridIsInit)
    {
        mParamGridIsInit = true;
        mStepGrid = aCam.mStepGrid;
        mRayonInvGrid = aCam.mRayonInvGrid;
    }
    mScanned = aCam.mScanned;
}

void  ElCamera::AddDistCompl
      (
          const std::vector<bool> &                   aVD,
          const std::vector<ElDistortion22_Gen *> &   aV
      )
{
    for (int aK=0 ; aK<int(aV.size()) ; aK++)
    {
        AddDistCompl(aVD[aK],aV[aK]);
    }
}

const ElDistortion22_Gen   &  ElCamera::Get_dist() const
{
   return Dist();
}
ElDistortion22_Gen   &  ElCamera::Get_dist()
{
   return Dist();
}


const std::vector<ElDistortion22_Gen *> & ElCamera::DistCompl() const
{
  return mDistCompl;
}

const std::vector<bool> & ElCamera::DistComplIsDir() const
{
  return mDComplIsDirect;
}


Pt2dr ElCamera::DComplC2M(Pt2dr aP,bool UseTrScN) const
{

   aP = mGlobOrImaC2M(aP);
    if (mCRAP)
      aP = mCRAP->CorrC2M(aP);

   for (int aK=0 ; aK<int(mDistCompl.size()) ; aK++)
   {
        aP = mDComplIsDirect[aK]         ?
         mDistCompl[aK]->Inverse(aP) :
         mDistCompl[aK]->Direct(aP)  ;
   }

   if (UseTrScN)
   {
      aP = Pt2dr
           (
               (aP.x-mTrN.x)/mScN,
               (aP.y-mTrN.y)/mScN
           );
   }
   return aP;
}
Pt2dr ElCamera::NormC2M(Pt2dr aP) const
{
   aP = mGlobOrImaC2M(aP);
   return Pt2dr
          (
               (aP.x-mTrN.x)/mScN,
               (aP.y-mTrN.y)/mScN
          );
}
Pt2dr ElCamera::NormM2C(Pt2dr aP) const
{
   aP =   Pt2dr
          (
               aP.x*mScN+mTrN.x,
               aP.y*mScN+mTrN.y
          );
   return mGlobOrImaM2C(aP);
}


Pt2dr  ElCamera::DComplM2C(Pt2dr aP,bool UseTrScN ) const
{
    if (UseTrScN)
    {
       aP.x = aP.x * mScN + mTrN.x;
       aP.y = aP.y * mScN + mTrN.y;
    }

    for (int aK=int(mDistCompl.size())-1 ; aK>=0 ; aK--)
    {
        aP =  mDComplIsDirect[aK]           ?
          mDistCompl[aK]->Direct(aP)    :
          mDistCompl[aK]->Inverse(aP)   ;
    }
    if (mCRAP)
      aP = mCRAP->CorrM2C(aP);
   return mGlobOrImaM2C(aP);
}


ElRotation3D &       ElCamera::Orient()       {return _orient;}
const ElRotation3D & ElCamera::Orient() const {return _orient;}

void ElCamera::SetOrientation(const ElRotation3D &ORIENT)
{

     _orient = ORIENT;
}

void ElCamera::AddToCenterOptical(const Pt3dr & aOffsetC)
{
    Pt3dr aC = _orient.inv().ImAff(Pt3dr(0,0,0)) + aOffsetC;
    ElRotation3D aOrient(aC,_orient.inv().Mat(),true);
    _orient = aOrient.inv();
}

void ElCamera::MultiToRotation(const ElMatrix<double> & aOffsetR)
{
    Pt3dr aC = _orient.inv().ImAff(Pt3dr(0,0,0));
    ElRotation3D aOrient(aC,aOffsetR.transpose()*_orient.inv().Mat(),true);
    _orient = aOrient.inv();
}

Pt2dr ElCamera::R3toC2(Pt3dr p) const
{
    return Proj().Proj(R3toL3(p));
}

Pt3dr ElCamera::R3toL3(Pt3dr aPR3) const
{
   return _orient.ImAff(aPR3);
}

Pt3dr ElCamera::L3toR3(Pt3dr aPL3) const
{
   return _orient.ImRecAff(aPL3);
}



typedef Pt2dr (ElProj32:: * TP)(Pt3dr) const;

Pt2dr ElCamera::R3toF2(Pt3dr p) const
{
    return DistDirecte(Proj().Proj(R3toL3(p)));
}


Pt2dr ElCamera::F2toC2(Pt2dr p) const
{
   return DistInverse(p);
}

Pt3dr    ElCamera::C2toDirRayonL3(Pt2dr p) const
{
   return Proj().DirRayon(DComplC2M(p));
}

Pt3dr    ElCamera::F2toDirRayonL3(Pt2dr p) const
{
   // std::cout << "CCCC " << euclid(p,DistDirecte(DistInverse(p))) << "\n";
   // std::cout << p <<  IsInZoneUtile(p) << "\n";

   return Proj().DirRayon(DistInverse(p));
}
Pt2dr    ElCamera::F2toPtDirRayonL3(Pt2dr p) const
{
    Pt3dr aD3 = F2toDirRayonL3(p);
    return Pt2dr(aD3.x,aD3.y);
}

Pt3dr    ElCamera::F2toDirRayonR3(Pt2dr p) const
{
   return  _orient.IRecVect(F2toDirRayonL3(p));
}
Pt3dr    ElCamera::C2toDirRayonR3(Pt2dr p) const
{
   return  _orient.IRecVect(C2toDirRayonL3(p));
}


Pt3dr ElCamera::DirRayonR3(const Pt2dr & aPIm) const
{
    return F2toDirRayonR3(aPIm);
}




Pt3dr   ElCamera::DirK() const
{
   return  _orient.IRecVect(Pt3dr(0,0,1));
}

ElCplePtsHomologues ElCamera::F2toPtDirRayonL3(const ElCplePtsHomologues & aCpl,ElCamera * aCam2)
{
   if (aCam2==0) aCam2 = this;
   return ElCplePtsHomologues
          (
               F2toPtDirRayonL3(aCpl.P1()),
               aCam2->F2toPtDirRayonL3(aCpl.P2()),
           aCpl.Pds()
      );
}



static bool  OkPt(const Pt2dr & aP, const Pt2di & aSz)
{
   return (aP.x>=0) && (aP.y>=0) && (aP.x<=aSz.x) && (aP.y<=aSz.y);
}

ElPackHomologue  ElCamera::F2toPtDirRayonL3(const ElPackHomologue & aPckIn,ElCamera * aCam2)
{
  Pt2di aSz1 = Sz();
  Pt2di aSz2 = aCam2->Sz();

   if (aCam2==0)
       aCam2 = this;
   ElPackHomologue aPckOut;
   for
   (
       ElPackHomologue::const_iterator itP=aPckIn.begin();
       itP!=aPckIn.end();
       itP++
   )
   {
      if ( (! OkPt(itP->P1(),aSz1))  || (! OkPt(itP->P2(),aSz2)))
      {
// MPD => il semble que SIFT genere des points 
          std::cout << "IM1 , P : " << itP->P1() << aSz1 << "\n";
          std::cout << "IM2 , P : " << itP->P2() << aSz2 << "\n";
          ELISE_ASSERT
          (
             OkPt(itP->P1(),aSz1) && OkPt(itP->P2(),aSz2),
             "Pt Out Cam in ElCamera::F2toPtDirRayonL3"
          );
      }
      else
      {
           if (
                     IsInZoneUtile(itP->P1())
                  && aCam2->IsInZoneUtile(itP->P2())
              )
           {
               aPckOut.Cple_Add(F2toPtDirRayonL3(itP->ToCple(),aCam2));
           }
           else
           {
           }
      }
   }

   return aPckOut;
}

Appar23    ElCamera::F2toPtDirRayonL3(const Appar23 & anAp)
{
    return Appar23(F2toPtDirRayonL3(anAp.pim),anAp.pter);
}

std::list<Appar23>  ElCamera::F2toPtDirRayonL3(const std::list<Appar23> & aLin)
{
   std::list<Appar23> aRes;

   for
   (
       std::list<Appar23>::const_iterator itA=aLin.begin();
       itA != aLin.end();
       itA++
   )
   {
      aRes.push_back(F2toPtDirRayonL3(*itA));
   }

   return  aRes;
}


Pt2dr  ElCamera::L3toC2(Pt3dr p) const
{
    return Proj().Proj(p);
}
Pt2dr  ElCamera::L3toF2(Pt3dr p) const
{
    return DistDirecte(Proj().Proj(p));
}


Pt2dr ElCamera::Radian2Pixel(const Pt2dr & aP) const
{
     Pt3dr aQ(aP.x,aP.y,1.0);
     return L3toF2(aQ);

}

Pt2dr ElCamera::Pixel2Radian(const Pt2dr & aP) const
{
    Pt3dr aQ =  F2toDirRayonL3(aP);
    return Pt2dr(aQ.x,aQ.y) / aQ.z;
}



// Pt2dr Radian2Pixel() const;


Pt2dr   ElCamera::PtDirRayonL3toF2(Pt2dr aP) const
{
     return DistDirecte(Proj().Proj(Pt3dr(aP.x,aP.y,1.0)));
}

void ElCamera::F2toRayonL3(Pt2dr aPF2,Pt3dr &aP0,Pt3dr & aP1) const
{

   Proj().Rayon(F2toC2(aPF2),aP0,aP1);

}

void ElCamera::F2toRayonR3(Pt2dr aPF2,Pt3dr &aP0,Pt3dr & aP1) const
{
   F2toRayonL3(aPF2,aP0,aP1);
   aP0 = L3toR3(aP0);
   aP1 = L3toR3(aP1);
}




Pt3dr  ElCamera::F2AndZtoR3(const Pt2dr & aPIm,double aZ) const
{
    Pt3dr aQ0,aQ1;
    F2toRayonR3(aPIm,aQ0,aQ1);
    double aLambda = (aZ-aQ0.z)/ (aQ1.z-aQ0.z);
    return aQ0 + (aQ1-aQ0) * aLambda;
}

ElSeg3D ElCamera::F2toRayonR3(Pt2dr aPF2) const
{
   Pt3dr  aQa,aQb;
   F2toRayonR3(aPF2,aQa,aQb);
   return ElSeg3D(aQa,aQb);
}


Pt3dr ElCamera::PtFromPlanAndIm(const cElPlan3D  & aPlan,const Pt2dr& aP) const
{
  ElSeg3D aSeg =  F2toRayonR3(aP);
  return aPlan.Inter(aSeg);
}


ElCamera::~ElCamera()
{
}

//    Cam = R1 M1
//    Cam = R2 M2
//    Monde2 = S1tOS2 (Monde1)
//    S2toS1 = S2.FromSys2This(
//  R2 =  R1 S2toS1 = R1 S1.FromSys2This(S2



//
//  Orientation Monde to Cam (CamStenope::CentreOptique() = _orient.IRecAff(Pt3dr(0,0,0))
//
//       p = D ( Pi (RSrci3Cam PSrc))
//       PCam = RSrc2Cam PSrc  PCam  = RCible2Cam PCible
//       PGeoC = S2G (PSrc)      PCible = G2C (PGeoC)
//        RCible2Cam PCible = PCam =  RSrc2Cam PSrc = RSrc2Cam S2G-1 G2C-1 PCible
//
//        RCible2Cam -1     = G2C S2G RSrc2Cam -1
//        RCam2Cible = G2C S2G R Cam2Src

void TestMatr(const char * aMes, ElMatrix<double> aM)
{
   ShowMatr(aMes,aM);
   ShowMatr("MtM",aM*aM.transpose());
}

void TestCamCHC(ElCamera & aCam)
{
    double aProf0=100;
    Pt2dr aPIm1(220,120);
    std::cout << " TTttCamm  " << aCam.ImEtProf2Terrain(aPIm1,aProf0+1) -  aCam.ImEtProf2Terrain(aPIm1,aProf0) << "\n";
}


// Dans cEq12Param.cpp
extern void AffinePose(ElCamera & aCam,const std::vector<Pt2dr> & aVIm,const std::vector<Pt3dr> & aVPts);

/*   
    0 = (p0 x + p1 y + p2 z + p3) - I (p8 x + p9 y + p10 z + p11)
    0 = (p4 x + p5 y + p6 z + p7) - J (p8 x + p9 y + p10 z + p11)
*/

#if (0)
#endif


void ElCamera::ChangeSys(const std::vector<ElCamera *> & aVCam, const cTransfo3D & aTransfo3D,bool ForceRot,bool AtGroundLevel)
{

    // Pour l'instant, pas encore ecrit la fonction  qui transform l'eq aux 12 param en  param physique ....
    //
    //if (! ForceRot)
    {
        for (int aK=0 ; aK<int(aVCam.size()); aK++)
        {
            std::vector<Pt2dr> aVIm;
            std::vector<Pt2dr> aVPhGr;
            std::vector<Pt3dr> aVSource;
            cEq12Parametre anEq12;
            int aNbXY = 5;
            int aNbProf  = 3;
            double aPropProf= 0.2;

            ElCamera & aCam = *(aVCam[aK]);
            Pt2dr aSzP = aCam.SzPixel();
            double aProfMoy = aCam.GetRoughProfondeur();
            int anIndCentre =-1;
 
            for (int aKp= -aNbProf ; aKp<= aNbProf ; aKp++)
            {
                if (AtGroundLevel || (aKp!=0))
                {
                    double aMul = aKp/double(aNbProf);
                    if (AtGroundLevel) 
                       aMul =  pow(1+aPropProf,aMul);
                    double aProf =  aProfMoy * aMul;
                    for (int aKx=0 ; aKx<= aNbXY ; aKx++)
                    {
                        for (int aKy=0 ; aKy<= aNbXY ; aKy++)
                        {
                            Pt2dr aPIm = aSzP.mcbyc(Pt2dr(aKx,aKy)/aNbXY);
                            Pt2dr aPPhgr = aCam.F2toPtDirRayonL3(aPIm);
                        // if (Test) aPPhgr = Pt2dr(aPPhgr.x*1.1 +0.05 * aPPhgr.y,aPPhgr.y*0.9) + Pt2dr(0.1,0.15)  ;
                            Pt3dr aPSource = aCam.ImEtProf2Terrain(aPIm,aProf) ;//   + Pt3dr(1e6,1e7,1e5);
                            if ((aKp==0) && (aKx==0) && (aKy==0))
                               anIndCentre = (int)aVSource.size();
                            aVPhGr.push_back(aPPhgr);
                            aVSource.push_back(aPSource);
                            aVIm.push_back(aPIm);
                        }
                    }
                }
            }

            // std::vector<Pt3dr> aVCible = aCible.FromGeoC(aSource.ToGeoC(aVSource));
            std::vector<Pt3dr> aVCible = aTransfo3D.Src2Cibl(aVSource); 
            Pt3dr aPCentreCible  = aVCible[anIndCentre];
            for (int aKP = 0 ; aKP<int(aVCible.size()) ; aKP++)
            {
                anEq12.AddObs(aVCible[aKP],aVPhGr[aKP],1.0);
            }

            if (ForceRot) 
            {
                std::pair<ElMatrix<double>,ElRotation3D> aPair = anEq12.ComputeOrtho();
                aCam.SetOrientation(aPair.second.inv());
                aCam.SetAltiSol(aPCentreCible.z);
                aCam.SetProfondeur(aCam.ProfondeurDeChamps(aPCentreCible));

                AffinePose(aCam,aVIm,aVCible);
            }
            else
            {
                std::pair<ElMatrix<double>,Pt3dr>  aTransfo = anEq12.ComputeNonOrtho();
                ElRotation3D aOriCam2Cible(aTransfo.second,aTransfo.first,false);
                aCam.SetOrientation(aOriCam2Cible.inv());
                aCam.SetAltiSol(aPCentreCible.z);
                aCam.SetProfondeur(aCam.ProfondeurDeChamps(aPCentreCible));
            }
        }
        return ;
    }


#if (0) // Ancienne facon a priori obsolete

    bool Test = false;
    std::vector<Pt3dr> aVCenterSrc;
    std::vector<Pt3dr> aPMoy;
    std::vector<bool>  aPMoyIsCalc;


    Pt2dr aPIm0(0,0);
    Pt3dr aPSrc0(0,0,0);
    Pt3dr aGeoC0(0,0,0);
    Pt3dr aCible0(0,0,0);
    double aProf0=100;


    Pt2dr aPIm1(0,0);
    Pt3dr aPSrc1(0,0,0);
    Pt3dr aGeoC1(0,0,0);
    Pt3dr aCible1(0,0,0);
    double aProf1=110;
/*
*/

    for (int aK=0 ; aK<int(aVCam.size()); aK++)
    {
        ElCamera & aCam = *(aVCam[aK]);

        if (aK==0)
        {
            aPIm0 = Pt2dr(aCam.Sz()) / 2.0;
            aPSrc0 = aCam.ImEtProf2Terrain(aPIm0,aProf0);
            aGeoC0 = aSource.ToGeoC(aPSrc0);
            aCible0  = aCible.FromGeoC(aGeoC0);

            aPIm1 = Pt2dr(aCam.Sz()) / 4.0;
            aPSrc1 = aCam.ImEtProf2Terrain(aPIm1,aProf1);
            aGeoC1 = aSource.ToGeoC(aPSrc1);
            aCible1  = aCible.FromGeoC(aGeoC1);


            if (Test)
            {
                std::cout << "Im=" << aPIm0 << " Src=" << aPSrc0 << " G="<<aGeoC0 << " Cbl=" <<  aCible0 << "\n";
                std::cout << "VEC ::  Src=" << aPSrc1-aPSrc0 << " G="<< aGeoC1- aGeoC0 << " Cbl=" << aCible1- aCible0 << "\n";
                std::cout << "REPROJ-init " << aCam.R3toF2(aPSrc0) << aCam.R3toF2(aPSrc1) << "\n";
            }
        }

        const ElRotation3D &  aOriSrc2Cam = aCam.Orient();
        Pt3dr aC = aOriSrc2Cam.ImRecAff(Pt3dr(0,0,0));
        aVCenterSrc.push_back(aC);
        if (aCam.ProfIsDef())
        {
           aPMoy.push_back(aCam.ImEtProf2Terrain(Pt2dr(aCam.Sz())/2.0,aCam.GetProfondeur()));
           aPMoyIsCalc.push_back(true);
        }
        else
        {
           aPMoy.push_back(aC);
           aPMoyIsCalc.push_back(false);
        }
    }

    double aEpsilon = 50.0;

    std::vector<Pt3dr> aVCenterGeoc;
    std::vector<ElMatrix<double> > aVJacS2G = aSource.Jacobien(aVCenterSrc,Pt3dr(aEpsilon,aEpsilon,aEpsilon),true,&aVCenterGeoc);

    std::vector<Pt3dr> aVCenterCible;
    std::vector<ElMatrix<double> > aVJacG2C = aCible.Jacobien(aVCenterGeoc,Pt3dr(aEpsilon,aEpsilon,aEpsilon),false,&aVCenterCible);

    aPMoy  = aCible.FromGeoC(aSource.ToGeoC(aPMoy));

    for (int aK=0 ; aK<int(aVCam.size()); aK++)
    {
        ElCamera & aCam = *(aVCam[aK]);

        ElRotation3D  aOriCam2Src = aCam.Orient().inv();
        ElMatrix<double> aMatCam2Cible  = aVJacG2C[aK] * aVJacS2G[aK] * aOriCam2Src.Mat();

// TestMatr("aVJacG2C ",  aVJacG2C[aK]);

        if (ForceRot)
        {
          aMatCam2Cible = NearestRotation(aMatCam2Cible);
        }

        ElRotation3D aOriCam2Cible(aVCenterCible[aK],aMatCam2Cible,false);

        aCam.SetOrientation(aOriCam2Cible.inv());


        if (aPMoyIsCalc[aK])
        {
           aCam.SetAltiSol(aPMoy[aK].z);
        }
        else
        {
           aCam.UndefAltisSol();
        }

        if (aK==0)
        {
            if (Test)
            {
                std::cout << "REPROJ-finale " << aCam.R3toF2(aCible0) << " " << aCam.R3toF2(aCible1)  << "\n";
                Pt3dr aPC0_Bis = aCam.ImEtProf2Terrain(aPIm0,aProf0);
                std::cout << "Inv Proj Finale " << aCam.R3toF2(aPC0_Bis) << " " << euclid(aCible0 - aPC0_Bis)  << "\n";
                std::cout << " VEC0 " << aCam.ImEtProf2Terrain(aPIm1,aProf0+1) -  aCam.ImEtProf2Terrain(aPIm1,aProf0) << "\n";
                TestCamCHC(aCam);
            }
        }
    }
#endif
}



REAL ElCamera::EcProj
     (
          const ElSTDNS list<Pt3dr> & PR3 ,
          const ElSTDNS list<Pt2dr> & PF2
     ) const
{
    ELISE_ASSERT((PR3.size() == PF2.size()),"size != in ElCamera::EcProj");

    REAL res = 0;

    ElSTDNS list<Pt3dr>::const_iterator It3 = PR3.begin();
    ElSTDNS list<Pt2dr>::const_iterator It2 = PF2.begin();

    while (It3 != PR3.end())
    {
        res += euclid(*It2,R3toF2(*It3));
        It3++; It2++;
    }
    return res;
}

bool  ElCamera::Devant(const Pt3dr & aPTer) const
{
    Pt3dr aPL = R3toL3(aPTer);
    return aPL.z > 0;
}

bool  ElCamera::TousDevant(const list<Pt3dr> & aL) const
{
   for (std::list<Pt3dr>::const_iterator itP=aL.begin(); itP!=aL.end() ; itP++)
       if (!Devant(*itP))
          return false;

   return true;
}


REAL ElCamera::EcProj(const ElSTDNS list<Appar23> & P23)
{
    ElSTDNS list<Pt3dr> PR3;
    ElSTDNS list<Pt2dr> PF2;

    ToSepList(PR3,PF2,P23);

    return EcProj(PR3,PF2);
}

void ElCamera::DiffR3F2(ElMatrix<REAL> & M,Pt3dr r) const
{

    Pt3dr l = _orient.ImAff(r);
    Pt2dr c = Proj().Proj(l);

    ELISE_ASSERT(mDIsDirect,"No ElCamera::DiffR3F2");
    ELISE_ASSERT(mDistCompl.empty(),"No ElCamera::DiffR3F2");
    M =  Dist().Diff(c) * Proj().Diff(l) *  _orient.Mat();

}
ElMatrix<REAL>  ElCamera::DiffR3F2(Pt3dr r) const
{
   ElMatrix<REAL> M(3,2);
   DiffR3F2(M,r);
   return M;
}

void ElCamera::DiffR3F2Param(ElMatrix<REAL> & M,Pt3dr r) const
{
    Pt3dr l = _orient.ImAff(r);
    Pt2dr c = Proj().Proj(l);

    ELISE_ASSERT(mDIsDirect,"No ElCamera::DiffR3F2");
    ELISE_ASSERT(mDistCompl.empty(),"No ElCamera::DiffR3F2");
    M =  Dist().Diff(c) * Proj().Diff(l) *  _orient.DiffParamEn1pt(r);
}

ElMatrix<REAL>  ElCamera::DiffR3F2Param(Pt3dr r) const
{
     ElMatrix<REAL> M(6,2);
     DiffR3F2Param(M,r);
     return M;
}

REAL ElCamera::EcartProj(Pt2dr aPF2A,Pt3dr aPR3,Pt3dr aDirR3) const
{
    Pt2dr aP1 = R3toF2(aPR3);
    Pt2dr aP2 = R3toF2(aPR3 + aDirR3/euclid(aDirR3));
    SegComp aSeg(aP1,aP2);

    return aSeg.dist(SegComp::droite,aPF2A);
}

REAL ElCamera::EcartProj(Pt2dr aPF2A,const ElCamera & CamB,Pt2dr aPF2B) const
{
    Pt3dr RayA0,RayA1;
    F2toRayonR3(aPF2A,RayA0,RayA1);
    ElSeg3D aSegA(RayA0,RayA1);


    Pt3dr RayB0,RayB1;
    CamB.F2toRayonR3(aPF2B,RayB0,RayB1);
    ElSeg3D aSegB(RayB0,RayB1);

    Pt3dr aPrA,aPrB;

    aSegA.Projections(aPrA,aSegB,aPrB);

    return         EcartProj(aPF2A,aPrB,RayB1-RayB0)
            + CamB.EcartProj(aPF2B,aPrA,RayA1-RayA0);
}

cOrientationConique  ElCamera::StdExportCalibGlob() const
{
   return StdExportCalibGlob(true);
}

std::string  ElCamera::StdExport2File(cInterfChantierNameManipulateur *anICNM,const std::string & aDirOri,const std::string & aNameIm,const std::string & aNameFileInterne)
{
   bool FileInterne = (aNameFileInterne != "");
   cOrientationConique  anOC = StdExportCalibGlob() ;
   if (FileInterne)
   {
      anOC.Interne().SetNoInit();
      anOC.FileInterne().SetVal(aNameFileInterne);
   }
   std::string aName = anICNM->NameOriStenope(aDirOri,aNameIm);
   MakeFileXML(anOC,aName);
   return aName;
}


cOrientationConique  ElCamera::StdExportCalibGlob(bool ModeMatr) const
{
   // std::cout << "PROFONDEUR " << mProfondeur << "\n";
   cOrientationConique aRes = ExportCalibGlob
          (
               Sz(),
               mAltiSol,
               mProfondeur,
               0,
               ModeMatr,
               "???hhh"
          );

   return aRes;
}



cVerifOrient ElCamera::MakeVerif(int aNbVerif,double aProf,const char * aNAux,const Pt3di * aVerifDet) const
{

   FILE * aFPAux = 0;

   if (aNAux)
   {
       std::string * aNS = new std::string(StdPrefix(aNAux)+".txt");
       aFPAux = ElFopen(aNS->c_str(),"w");
   }

   cVerifOrient aVerif;
   aVerif.Tol() = 1e-3;
   Box2dr aBU = BoxUtile();

   int aK=0;
   if (aVerifDet)
   {
       cListeAppuis1Im aLAp;
       for (int anX=0 ; anX<aVerifDet->x; anX++)
       {
           for (int anY=0 ; anY<aVerifDet->y; anY++)
           {
                for (int aZ=0 ; aZ<aVerifDet->z; aZ++)
                {
                      Pt2dr aP2  = aBU.FromCoordLoc(Pt2dr(anX/double(aVerifDet->x),anY/double(aVerifDet->y)));
                      if (IsInZoneUtile(aP2))
                      {
                         double aP =  1000.0 * (1+aZ);

                         Pt3dr aP3 = ImEtProf2Terrain(aP2,aP);
                         Pt2dr aQ2 = R3toF2(aP3);

                         cMesureAppuis aMA;
                         aMA.Im() = aQ2;
                         aMA.Ter() = aP3;
                         aMA.Num().SetVal(aK);
                         aK++;
                         aLAp.Mesures().push_back(aMA);
                         //aVerif.Appuis().push_back(aMA);
                      }

                }
           }
       }
       aVerif.AppuisConv().SetVal(aLAp);
   }

   for (aK=0 ; aK< aNbVerif ; )
   {
       Pt2dr aP2  = aBU.RandomlyGenereInside();
       if (IsInZoneUtile(aP2))
       {
// std::cout << "P2 = " << aP2 << "\n";
          double aP = (0.5 +  NRrandom3()) * aProf;
          Pt3dr aP3 = ImEtProf2Terrain(aP2,aP);
          Pt2dr aQ2 = R3toF2(aP3);

       // std::cout << euclid(aQ2,aP2) << " :: " << aQ2 << aP2 << "\n";
          cMesureAppuis aMA;
          aMA.Im() = aQ2;
          aMA.Ter() = aP3;
          aMA.Num().SetVal(aK);
          aVerif.Appuis().push_back(aMA);

          if (aFPAux)
          {
             Pt3dr aL3 =  R3toL3(aP3);
             Pt2dr aC2  = R3toC2(aP3);
             Pt2dr aF2  = R3toF2(aP3);

             fprintf(aFPAux,"NUM%d\n",aK);
             fprintf(aFPAux,"L3 %lf %lf %lf\n",aL3.x,aL3.y,aL3.z);
             fprintf(aFPAux,"C2 %lf %lf\n",aC2.x,aC2.y);
             fprintf(aFPAux,"F2 %lf %lf\n",aF2.x,aF2.y);
/*
             const cCamStenopeDistRadPol * aCDR = Debug_CSDRP() ;
             if (aCDR)
             {
                  const ElDistRadiale_PolynImpair & aPDR = aCDR->DRad();

                  double aR = euclid(aPDR.Centre()-aC2);
                  fprintf(aFPAux,"RAY %lf DIST %lf\n",aR,aPDR.DistDirecte(aR) * aR);
                  fprintf(aFPAux,"RAY - MAX %lf\n", aPDR.RMax());
             }
*/
             fprintf(aFPAux,"\n");
          }
          aK++;
       }
   }

   if (aFPAux)
      ElFclose(aFPAux);

   return aVerif;
}

cOrientationConique  ElCamera::ExportCalibGlob
                     (
                         Pt2di aSzIm,
                         double AltiSol,
                         double Prof,
                         int aNbVerif,
                         bool aModeMatr,
                         const char * aNameAux,
                         const Pt3di * aNbVeridDet
                      ) const
{
   cCalibrationInternConique aCIC = ExportCalibInterne2XmlStruct(aSzIm);
   cOrientationExterneRigide anOER = From_Std_RAff_C2M(_orient.inv(),aModeMatr);
   anOER.AltiSol().SetVal(AltiSol);
   if (Prof >0)
      anOER.Profondeur().SetVal(Prof);
   else
      anOER.Profondeur().SetNoInit();


    if (VitesseIsInit())
       anOER.Vitesse().SetVal(Vitesse());
   anOER.IncCentre().SetVal(IncCentre());

   cOrientationConique anOC;
   anOER.Time().SetVal(GetTime());
   anOC.Interne().SetVal(aCIC);
   anOC.Externe() = anOER;
   anOC.ConvOri().KnownConv().SetVal(DistIsDirecte() ? eConvApero_DistM2C : eConvApero_DistC2M);

   anOC.OrIntImaM2C().SetVal(El2Xml(mScanOrImaM2C));
   anOC.TypeProj().SetVal(El2Xml(mTypeProj));

   anOC.ZoneUtileInPixel().SetVal(GetZoneUtilInPixel());

   if (aNbVerif || aNbVeridDet)
   {
      anOC.Verif().SetVal(MakeVerif(aNbVerif,Prof,aNameAux,aNbVeridDet));
   }

   return anOC;

}

cCalibrationInternConique  ElCamera::ExportCalibInterne2XmlStruct(Pt2di aSzIm) const
{
   cCalibrationInternConique aParam;
   /*
   ELISE_ASSERT(!mDIsDirect,"XML sens M2C non supporte");
   */

    aParam.KnownConv().SetVal(DistIsDirecte() ? eConvApero_DistM2C : eConvApero_DistC2M);
    InstanceModifParam(aParam);
    // aParam.PP() = PP();
    // aParam.F()  = Focale();
    aParam.SzIm() = aSzIm;
    if (mSzPixel.x >0)
      aParam.PixelSzIm().SetVal(mSzPixel);


    aParam.OrIntGlob().SetNoInit();
    if (! mIntrOrImaC2M.IsId())
    {
        cOrIntGlob anOIG;
        anOIG.Affinite() = El2Xml(mIntrOrImaC2M);
        anOIG.C2M() = true;
        aParam.OrIntGlob().SetVal(anOIG);
    }
    if (HasRayonUtile())
    {
       aParam.RayonUtile().SetVal(RayonUtile());
    }
    if (mParamGridIsInit)
    {
       cParamForGrid aPFG;
       aPFG.StepGrid() = mStepGrid;
       aPFG.RayonInv() = mRayonInvGrid;
       aParam.ParamForGrid().SetVal(aPFG);
    }

    bool OneDif = false;
    for (int aKD=0 ; aKD<int(mDistCompl.size()) ; aKD++)
    {
        bool  isKD = mDComplIsDirect[aKD];
        aParam.CalibDistortion().push_back(mDistCompl[aKD]->ToXmlStruct(this));
    aParam.ComplIsC2M().push_back(! isKD);
    if (isKD!=DistIsDirecte())
       OneDif = true;
    }

    if (!OneDif)
        aParam.ComplIsC2M().clear();

    aParam.CalibDistortion().push_back(Get_dist().ToXmlStruct(this));

    if (mCRAP)
    {
       aParam.CorrectionRefractionAPosteriori().SetVal(mCRAP->ToXML());
    }


    if (IsScanned())
       aParam.ScannedAnalogik().SetVal(true);

   return aParam;
}

/***************************************************************/
/*                                                             */
/*              CamStenope                                     */
/*                                                             */
/***************************************************************/

CamStenope * CamStenope::DownCastCS() { return this; }

std::string CamStenope::Save2XmlStdMMName
     (
           cInterfChantierNameManipulateur * anICNM,
           const std::string & aOriOut,
           const std::string & aNameImClip,
           const ElAffin2D & anOrIntInit2Cur
     ) const 
{
    cOrientationConique  aCO = StdExportCalibGlob();
    std::string aNameOut =  anICNM->Dir() + anICNM->NameOriStenope(aOriOut,aNameImClip);
    cCalibrationInternConique * aCIO = aCO.Interne().PtrVal();
    ELISE_ASSERT(aCIO!=0,"cCalibrationInternConique in CamStenope::Save2XmlStdMMName");
    // cCalibrationInterneRadiale * aMR =aCIO->CalibDistortion().back().ModRad().PtrVal();
    ElAffin2D aM2C0 = Xml2EL(aCO.OrIntImaM2C());
    ElAffin2D  aM2CCliped =    anOrIntInit2Cur.inv() * aM2C0;
    aCO.OrIntImaM2C().SetVal(El2Xml(aM2CCliped));
    aCO.Interne().Val().PixelSzIm().SetVal(Pt2dr(Tiff_Im::UnivConvStd(aNameImClip).sz()));

    MakeFileXML(aCO,aNameOut);

    return aNameOut;
}

double  CamStenope::GetRoughProfondeur() const
{
    if (ProfIsDef())  return mProfondeur;
    if (AltisSolIsDef()) return ElAbs(PseudoOpticalCenter().z-mAltiSol);

    ELISE_ASSERT(false,"Nor Prof nor Alti in ElCamera::GetRoughProfondeur");
    return 0;
}




Pt3dr CamStenope::OrigineProf() const
{
    return PseudoOpticalCenter();
}


bool  CamStenope::HasOrigineProf() const
{
    return true;
}



double   ElCamera::EcartAngulaire(const Appar23 & anAp) const
{
    Pt3dr aD1 =  vunit(R3toL3(anAp.pter));
    Pt3dr aD2 =  vunit(F2toDirRayonL3(anAp.pim));

    return ElAbs(acos(scal(aD1,aD2)));
}

double   ElCamera::SomEcartAngulaire(const std::vector<Appar23> & aVApp) const
{
   double aSom = 0;
   for (int aK=0 ; aK<int(aVApp.size()) ; aK++)
       aSom += EcartAngulaire(aVApp[aK]);

   return aSom;
}



double   ElCamera::EcartAngulaire
        (
             Pt2dr aPF2A,
             const ElCamera & CamB,
             Pt2dr aPF2B
        ) const
{
   ElSeg3D aSegA = F2toRayonR3(aPF2A);
   ElSeg3D aSegB = CamB.F2toRayonR3(aPF2B);

   Pt3dr  aRes = aSegA.PseudoInter(aSegB);
   double aDist = aSegA.DistDoite(aRes);

   return   ElAbs(atan2(euclid(aRes-OrigineProf()),aDist))
          + ElAbs(atan2(euclid(aRes-CamB.OrigineProf()),aDist));
}

double   ElCamera::SomEcartAngulaire
         (
             const ElPackHomologue & aPackH,
             const ElCamera & CamB,
             double & aSomP
         ) const
{
    aSomP=0;
    double aSomE=0;
    for
    (
        ElPackHomologue::const_iterator itC = aPackH.begin();
        itC != aPackH.end();
        itC++
    )
    {
       double aPds = itC->Pds();
       double anEc = EcartAngulaire(itC->P1(),CamB,itC->P2());
       aSomP += aPds;
       aSomE+= anEc *aPds;
    }
    return aSomE;
}

Pt3dr  ElCamera::PseudoInterPixPrec
       (
           Pt2dr aPF2A,
           const ElCamera & CamB,
           Pt2dr aPF2B,
           double & aD
       ) const
{
    Pt3dr aRes = PseudoInter(aPF2A,CamB,aPF2B);
    aD  =  (euclid(Ter2Capteur(aRes)-aPF2A) + euclid(CamB.Ter2Capteur(aRes)-aPF2B))/2.0;

    return aRes;
}



Pt3dr   ElCamera::PseudoInter
        (
         Pt2dr aPF2A,
         const ElCamera & CamB,
         Pt2dr aPF2B,
         double * aDist
    ) const
{
   ElSeg3D aSegA = F2toRayonR3(aPF2A);
   ElSeg3D aSegB = CamB.F2toRayonR3(aPF2B);

   Pt3dr aRes = aSegA.PseudoInter(aSegB);
   if (aDist)
       *aDist = aSegA.DistDoite(aRes);
   return aRes;
}

Pt3dr  ElCamera::CdgPseudoInter(const ElPackHomologue & aPckIn,const ElCamera & CamB,double & aD) const
{
   double aSPds =0;
   Pt3dr aSInter (0,0,0);
   aD = 0;
   for
   (
       ElPackHomologue::const_iterator itP=aPckIn.begin();
       itP!=aPckIn.end();
       itP++
   )
   {
      aSPds += itP->Pds();
      double aD0;
      Pt3dr aP0 = PseudoInter(itP->P1(),CamB,itP->P2(),&aD0);
      // std::cout << aP0 << aD0 << itP->P1() << itP->P2() << "\n";
      aSInter = aSInter + aP0 *itP->Pds();
      aD += aD0*itP->Pds();
   }
   ELISE_ASSERT(aSPds!=0,"ElCamera::CdgPseudoInter No Points !!");
   aD /= aSPds;
   return aSInter / aSPds;
}



/*************************************************/
/*                                               */
/*    CamStenope                                 */
/*                                               */
/*************************************************/

double CamStenope::ResolutionSol() const
{
    return ResolutionAngulaire() * GetProfondeur();
}

double CamStenope::ResolutionSol(const Pt3dr & aP) const
{
    return ResolutionAngulaire() * ProfondeurDeChamps(aP);
}



double CamStenope::ResolutionAngulaire() const
{
   double aEps = 1e-3;
   // double aEps = 1e-5;
/*  Ne marche pas avec les cameras de type epipolair quand la vertical local est hor image
   double aEps = 1e-5;
   Pt2dr aP0 = L3toF2(Pt3dr(0,0,1));
   Pt2dr aP1 = L3toF2(Pt3dr(aEps,0,1));

   double aD = euclid(aP0-aP1) / aEps;

   return 1/aD ;
   return  _orient.IRecVect(F2toDirRayonL3(p));
*/



   // Ci dessus ne marche pas avec point hors image
// std::cout << "Szzz " << Sz() << "\n"; getchar();
   // Pt2dr aMil = Pt2dr(Sz()) /2.0;

   // Pt2dr aMil = DComplM2C(Pt2dr(Sz()) /2.0);
   Pt2dr aMil = SzPixel()/2.0;
   Pt2dr aMil2 = aMil + Pt2dr(aEps,0);
// aMil = DComplM2C(aMil);

   // std::cout << "Dif MIL = " << aMil -aMil2 << "\n";
   // std::cout << "Dif Ray = " << F2toDirRayonR3(aMil) -F2toDirRayonR3(aMil2) << "\n";
   // std::cout << "Dif Ray = " << F2toDirRayonL3(aMil) -F2toDirRayonL3(aMil2) << "\n";

   // Pt3dr aQ0 = ImEtProf2Terrain(aMil,1.0);    NE MARCHE PAS AVEC GRDE COORDONNEES SUR LES CENTRES
   // Pt3dr aQ1 = ImEtProf2Terrain(aMil2,1.0);


   Pt3dr aQ0 = F2toDirRayonL3(aMil);
   Pt3dr aQ1 = F2toDirRayonL3(aMil2);

   double aDQ = euclid(aQ0-aQ1) / aEps;


/*
   std::cout << mGlobOrImaC2M(aMil)  <<  mGlobOrImaC2M(aMil2) << "\n";
   std::cout << DComplC2M(aMil)  <<  DComplC2M(aMil2) << "\n";
   std::cout << DistInverse(aMil)  <<  DistInverse(aMil2) << "\n";
   std::cout << " dddDQ " <<  aMil << " " << aDQ  << aQ0 << aQ1 << "\n"; getchar();
*/


    return aDQ;
/*
*/

// std::cout << "DdddddddDD  " << aD << " " << aD * aDQ<< "\n";
// getchar();

/*
La focale ne marche pas avec les grille tres loin de Id
   std::cout << "RAGgg " << aD << " " << 1/aD << "\n";
getchar();

   return  1 / Focale();
*/
}

REAL CamStenope::Focale() const
{
   return _PrSten.focale();
}

Pt2dr CamStenope::PP() const
{
    return _PrSten.PP();
}

void CamStenope::Coins(Pt3dr &aP1,Pt3dr &aP2,Pt3dr &aP3,Pt3dr &aP4, double aZ) const
{
    aP1 = ImEtProf2Terrain(Pt2dr(0.f,0.f),aZ);       // HAUT GAUCHE
    aP2 = ImEtProf2Terrain(Pt2dr(Sz().x,0.f),aZ);    // HAUT DROIT
    aP3 = ImEtProf2Terrain(Pt2dr(0.f,Sz().y),aZ);    // BAS GAUCHE
    aP4 = ImEtProf2Terrain(Pt2dr(Sz().x,Sz().y),aZ); // BAS DROIT
}

// for  aerial imagery, project the 4 camera corners on a ground surface assumed to be at Z=aZ
void CamStenope::CoinsProjZ(Pt3dr &aP1,Pt3dr &aP2,Pt3dr &aP3,Pt3dr &aP4, double aZ) const
{
    aP1 = ImEtZ2Terrain(Pt2dr(0.f,0.f),aZ);       // HAUT GAUCHE
    aP2 = ImEtZ2Terrain(Pt2dr(Sz().x,0.f),aZ);    // HAUT DROIT
    aP3 = ImEtZ2Terrain(Pt2dr(0.f,Sz().y),aZ);    // BAS GAUCHE
    aP4 = ImEtZ2Terrain(Pt2dr(Sz().x,Sz().y),aZ); // BAS DROIT
}
// return ground box
Box2dr CamStenope::BoxTer(double aZ) const
{
    Pt3dr aP1,aP2,aP3,aP4;
    CoinsProjZ(aP1,aP2,aP3,aP4,aZ);
    Pt2dr aPMin = Pt2dr(ElMin(aP1.x,ElMin(aP2.x,ElMin(aP3.x,aP4.x))),ElMin(aP1.y,ElMin(aP2.y,ElMin(aP3.y,aP4.y))));
    Pt2dr aPMax = Pt2dr(ElMax(aP1.x,ElMax(aP2.x,ElMax(aP3.x,aP4.x))),ElMax(aP1.y,ElMax(aP2.y,ElMax(aP3.y,aP4.y))));
    return Box2dr(aPMin,aPMax);
}

void ElCamera::SetSzPixel(const Pt2dr & aSzP)
{
   mSzPixel = aSzP;
}

Pt2dr  ElCamera::SzPixelBasik() const
{
   return mSzPixel;
}

Pt2dr  ElCamera::SzPixel() const
{
   if (mSzPixel.x>0)
      return mSzPixel;

   return DComplM2C(Pt2dr(Sz()) /2.0,false) * 2;
}

double ElCamera::ProfondeurDeChamps(const Pt3dr & aP) const
{
   return scal(DirVisee(),aP-OrigineProf());
}

Pt3dr ElCamera::DirVisee() const
{
    return _orient.IRecVect(Pt3dr(0,0,1));
}


const tParamAFocal & CamStenope::ParamAF() const
{
   return _PrSten.ParamAF();
}


Pt3dr CamStenope::Im1DirEtProf2_To_Terrain
      (
           Pt2dr aPIm,
           const CamStenope &  ph2,
           double prof2,
           const Pt3dr & aDir
      ) const
{
   Pt3dr  aRay =  F2toDirRayonR3(aPIm);
   Pt3dr  aC1 = OpticalVarCenterIm (aPIm);
   Pt3dr  aC2 = ph2.OrigineProf ();

   double aLamda =  (prof2+scal(aC2-aC1,aDir))/scal(aRay,aDir);
   return aC1 + aRay * aLamda;
}


Pt3dr CamStenope::Im1EtProfSpherik2_To_Terrain
      (
           Pt2dr aPIm,
           const CamStenope &  ph2,
           double prof2
      ) const
{
   Pt3dr  aRay =  F2toDirRayonR3(aPIm);
   Pt3dr  aC1 = OpticalVarCenterIm (aPIm);
   Pt3dr  aC2 = ph2.OrigineProf ();
   Pt3dr aV12 = aC1 - aC2;

   double A = square_euclid(aRay);
   double B = 2 * scal(aV12,aRay);
   double C = square_euclid(aV12) - ElSquare(prof2);

   double aDelta = ElMax(B*B - 4*A*C,0.0);

   double aLamda =  (-B+sqrt(aDelta))/ (2*A);
   return aC1 + aRay * aLamda;
}









Pt3dr  CamStenope::ImEtProfSpherik2Terrain
       (
            const Pt2dr & aPIm,
            const REAL & aProf
        ) const
{
    Pt3dr  aRay =  F2toDirRayonR3(aPIm);

    return    OpticalVarCenterIm(aPIm) +  vunit(aRay) * aProf;
}


Pt3dr  CamStenope::ImDirEtProf2Terrain
       (
            const Pt2dr & aPIm,
            const REAL & aProf,
            const Pt3dr & aNormPl
        ) const
{
    Pt3dr  aRay =  F2toDirRayonR3(aPIm);
    double aLamda = aProf / scal(aRay,aNormPl);

    return    OpticalVarCenterIm(aPIm) +  aRay * aLamda;
}





double CamStenope::ProfInDir(const Pt3dr & aP,const Pt3dr & aDir) const
{
    return scal(aP-OpticalVarCenterTer (aP),aDir);
}



Pt3dr  CamStenope::ImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
     return OpticalVarCenterIm(aP) + F2toDirRayonR3(aP) * aZ;
}

Pt3dr  CamStenope::NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
     return OpticalVarCenterIm(aP) + C2toDirRayonR3(aP) * aZ;
}




Pt3dr  CamStenope::ImEtZ2Terrain(const Pt2dr & aP,double aZ) const
{
//  std::cout << aP << Focale() << PP () << "\n";

     Pt3dr aC = OpticalVarCenterIm(aP);
     Pt3dr aRay =  F2toDirRayonR3(aP);

     double aLamda =  (aZ-aC.z)/aRay.z;

     return aC +  aRay * aLamda;
}







Ori3D_Std *  CamStenope::CastOliLib()
{
   return 0;
}

Ori3D_Std *  CamStenope::NN_CastOliLib()
{
   Ori3D_Std * aRes = CastOliLib();
   ELISE_ASSERT(aRes!=0,"CamStenope::NN_CastOliLib");

   return aRes;
}
double CamStenope::ResolutionPDVVerticale()
{
//    std::cout <<  "fffFOCALE " << Focale() <<  " " << 1/ ResolutionAngulaire()  << "\n";
// getchar();
  // return (PseudoOpticalCenter().z -GetAltiSol()) / Focale();
  return (PseudoOpticalCenter().z -GetAltiSol()) * ResolutionAngulaire() ;
}


const cCamStenopeDistRadPol * CamStenope::Debug_CSDRP() const
{
   return 0;
}

bool CamStenope::CanExportDistAsGrid() const
{
    return true;
}

CamStenope * CamStenope::Dupl() const
{
  return NS_ParamChantierPhotogram::Cam_Gen_From_XML(StdExportCalibGlob(),0,IdentCam())->CS();
}


CamStenope * CamStenope::StdCamFromFile
             (
                     bool CanUseGr,
                     const std::string & aName,
                     cInterfChantierNameManipulateur * anICNM
             )
{
  if ( ERupnik_MM() ) // NIKRUP
     std::cout << "NAME StdCamFromFile " << aName << "\n";

  return Gen_Cam_Gen_From_File(CanUseGr,aName,"OrientationConique",anICNM)->CS();
}



Pt3dr CamStenope::PseudoOpticalCenter() const
{
    return _orient.ImRecAff(Pt3dr(0,0,0));
}

Pt3dr    CamStenope::OpticalCenterOfPixel(const Pt2dr & aP) const 
{
   return PseudoOpticalCenter();
}

bool CamStenope::UseAFocal() const
{
     return mUseAF;
}

Pt3dr CamStenope::VraiOpticalCenter() const
{
    ELISE_ASSERT
    (
       ! mUseAF,
       "VRAIE STENOPE REQUIRED"
    );
    return PseudoOpticalCenter();
}

Pt3dr CamStenope::OpticalVarCenterIm(const Pt2dr & aPIm) const
{
   if (!mUseAF) return PseudoOpticalCenter();
   return _orient.ImRecAff(_PrSten.CentreProjIm(F2toC2(aPIm)));
}

Pt3dr CamStenope::OpticalVarCenterTer(const Pt3dr & aP) const
{
   if (!mUseAF) return PseudoOpticalCenter();
   return _orient.ImRecAff(_PrSten.CentreProjTer(_orient.ImAff(aP)));
}
/*
*/


ElProj32 & CamStenope::Proj()
{
   return _PrSten;
}

const ElProj32 & CamStenope::Proj() const
{
   return _PrSten;
}

ElDistortion22_Gen & CamStenope::Dist()
{
   ELISE_ASSERT(mDist!=0,"CamStenope::Dist()");
   return *mDist;
}

const ElDistortion22_Gen & CamStenope::Dist() const
{
   ELISE_ASSERT(mDist!=0,"CamStenope::Dist()");
   return *mDist;
}




/*************************************************/
/*                                               */
/*         cDistStdFromCam                       */
/*                                               */
/*************************************************/

cDistStdFromCam::cDistStdFromCam
(
   ElCamera & aCam
)  :
   mCam(aCam)
{
}

Pt2dr cDistStdFromCam::Direct(Pt2dr aP) const
{
     return mCam.F2toPtDirRayonL3(aP);
}
void  cDistStdFromCam::Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
{
     DiffByDiffFinies(aMat,aP,mCam.SzDiffFinie());
}



/*************************************************/
/*                                               */
/*    CamStenope                                 */
/*                                               */
/*************************************************/

CamStenope::CamStenope(bool isDistC2M,REAL Focale,Pt2dr centre,const std::vector<double>  & AFocalParam) :
    ElCamera  (isDistC2M,eProjectionStenope),
    _PrSten   (Focale,centre,AFocalParam),
    mUseAF    (_PrSten.UseAFocal()),
    mDist     (0)
{
}

void CamStenope::UnNormalize()
{
/*
   bool doScale =  Dist().AcceptScaling();
   bool doTr =     Dist().AcceptTranslate() ;
   std::cout << "UN-S " << doScale << " Tr " << doTr << "\n";
*/

   if ((mScN==1.0) && (mTrN==Pt2dr(0,0)))
     return;


   _PrSten.focale() *= mScN;
   _PrSten.SetPP(PP()*mScN + mTrN);

   Dist().SetScalingTranslate(1/mScN,-mTrN/mScN);

   mScN = 1.0;
   mTrN = Pt2dr(0,0);

}

void CamStenope::StdNormalise(bool doScale,bool  doTr,double aS,Pt2dr aTr)
{
   doScale = doScale && Dist().AcceptScaling();
   doTr =    doTr && Dist().AcceptTranslate() ;

   mScN =  doScale ? aS: 1.0;
   mTrN    =  doTr    ?  aTr : Pt2dr(0,0);


   if ((mScN==1.0) && (mTrN==Pt2dr(0,0)))
     return;

// std::cout << "IN " << PP() <<  " " << Focale() << "\n";
   _PrSten.focale() /= mScN;
   _PrSten.SetPP((PP()-mTrN) / mScN);
// std::cout << "OUT " << PP() <<  " " << Focale() << "\n";
   Dist().SetScalingTranslate(mScN,mTrN);
}


void CamStenope::StdNormalise(bool doScale,bool  doTr)
{
     StdNormalise(doScale,doTr,_PrSten.focale(),_PrSten.PP() );
}



double CamStenope::SzDiffFinie() const
{
    return Focale() / 1000.0;
}

CamStenope::CamStenope(const CamStenope & cam,const ElRotation3D & ORIENT) :
    ElCamera  (DistIsC2M(),eProjectionStenope),
    _PrSten   (cam._PrSten),
    mDist     (const_cast<ElDistortion22_Gen *>(&cam.Get_dist()))
{
    AddDistCompl(cam.DistComplIsDir(),cam.DistCompl());
    SetOrientation(ORIENT);
}

/*
CamStenope::CamStenope(const CamStenope & cam) :
    ElCamera  (DistIsC2M(),eProjectionStenope),
    _PrSten   (cam._PrSten),
    mDist     (const_cast<ElDistortion22_Gen *>(&cam.Get_dist()))
{
    AddDistCompl(cam.DistComplIsDir(),cam.DistCompl());
    SetOrientation(cam._orient);
    HeritComplAndSz(cam);
}
*/



void CamStenope::InstanceModifParam(cCalibrationInternConique & aParam)  const
{
    aParam.PP() = PP();
    aParam.F()  = Focale();
    aParam.ParamAF() = ParamAF() ;
}









void CamStenope::OrientFromPtsAppui
     (
         ElSTDNS list<ElRotation3D> & Res,
         Pt3dr R3A, Pt3dr R3B, Pt3dr R3C,
         Pt2dr F2A, Pt2dr F2B, Pt2dr F2C
     )
{

    Pt2dr C2A =  DistInverse(F2A);
    Pt2dr C2B =  DistInverse(F2B);
    Pt2dr C2C =  DistInverse(F2C);

    Pt3dr RayA =  _PrSten.DirRayon(C2A);
    Pt3dr RayB =  _PrSten.DirRayon(C2B);
    Pt3dr RayC =  _PrSten.DirRayon(C2C);





    ElSTDNS list<Pt3dr>  Prof;
    ElPhotogram::ProfChampsFromDist
    (
        Prof,
        RayA,RayB,RayC,
        euclid(R3A-R3B),euclid(R3A-R3C),euclid(R3B-R3C)
    );


    Res.clear();
    for (INT sign =-1; sign<=1 ; sign += 2)
    {
        for (ElSTDNS list<Pt3dr>::iterator it=Prof.begin() ; it!=Prof.end() ; it++)
        {
           Pt3dr L3A = RayA * it->x * sign;
           Pt3dr L3B = RayB * it->y * sign;
           Pt3dr L3C = RayC * it->z * sign;

           Pt3dr LAB = L3B-L3A;
           Pt3dr LAC = L3C-L3A;
           Pt3dr RAB = R3B-R3A;
           Pt3dr RAC = R3C-R3A;

           ElMatrix<REAL> M3 = MatFromImageBase
                               (
                                   RAB,RAC,RAB^RAC,
                                   LAB,LAC,LAB^LAC
                               );
           Res.push_back(ElRotation3D(L3A-M3*R3A,M3,true));
        }
    }
}


void CamStenope::OrientFromPtsAppui
     (
         ElSTDNS list<ElRotation3D> & Res,
         const ElSTDNS list<Pt3dr> & PR3 ,
         const ElSTDNS list<Pt2dr> & PF2
     )
{
     Res.clear();

     ELISE_ASSERT
     (
         (PR3.size()>=4) && (PR3.size() == PF2.size()),
         "Bad Size in OrientFromPtsAppui"
     );

    ElSTDNS list<Pt3dr>::const_iterator It3 = PR3.begin();
    ElSTDNS list<Pt2dr>::const_iterator It2 = PF2.begin();

    Pt3dr R3A = *It3; It3++;
    Pt3dr R3B = *It3; It3++;
    Pt3dr R3C = *It3; It3++;
    Pt2dr F2A = *It2; It2++;
    Pt2dr F2B = *It2; It2++;
    Pt2dr F2C = *It2; It2++;

    OrientFromPtsAppui(Res,R3A,R3B,R3C,F2A,F2B,F2C);
}

void CamStenope::OrientFromPtsAppui
     (
         ElSTDNS list<ElRotation3D> & Res,
         const ElSTDNS list<Appar23> & P23
     )
{
    ElSTDNS list<Pt3dr> PR3;
    ElSTDNS list<Pt2dr> PF2;

    ToSepList(PR3,PF2,P23);
    OrientFromPtsAppui(Res,PR3,PF2);
}


ElRotation3D  CamStenope::OrientFromPtsAppui
              (
                    bool RequireTousDevant,
                    const ElSTDNS list<Pt3dr> & PR3 ,
                    const ElSTDNS list<Pt2dr> & PF2 ,
                    REAL * Res_Dmin,
                    INT  * NbSol
              )
{

    ElSTDNS list<ElRotation3D>  Ors;
    OrientFromPtsAppui(Ors,PR3,PF2);


    if (NbSol)
    {
        *NbSol = (int) Ors.size();
    if (*NbSol == 0)
    {
            if (Res_Dmin)
              *Res_Dmin = 1e15;
            return ElRotation3D(Pt3dr(1000,2220,9990),10,20,30);
    }
    }
    else
    {
        ELISE_ASSERT(Ors.size()!=0,"Inc (No Solution) in OrientFromPtsAppui");
    }

    ElSTDNS list<ElRotation3D>::iterator it=Ors.begin();
    ElRotation3D res = *it;
    REAL dmin = 1e5;

    for (;it!=Ors.end();it++)
    {
       CamStenope Cam(*this,*it);
       if ((!RequireTousDevant) || Cam.TousDevant(PR3))
       {
          REAL dist = Cam.EcProj(PR3,PF2);

// if (DebugOFPA) std::cout << " " << dist << " " <<  it->IRecVect(Pt3dr(0,0,1)) << "\n";

          if (dist<dmin)
          {
              dmin = dist;
              res = *it;
          }
       }
    }

// if (DebugOFPA) std::cout <<  "\n";

    if (Res_Dmin)
       *Res_Dmin = dmin;

    return res;
}

void RansacTriplet
     (
         int & aK1,
         int & aK2,
         int & aK3,
     int   aNb
     )
{
   aK1 = NRrandom3(aNb);
   aK2 = NRrandom3(aNb);
   while (aK2==aK1)
   {
      aK2 = NRrandom3(aNb);
   }
   aK3 = NRrandom3(aNb);
   while ((aK3==aK1)  || (aK3==aK2))
   {
      aK3 = NRrandom3(aNb);
   }
}


ElRotation3D  CamStenope::CombinatoireOFPAGen
              (
                    bool TousDevant,
                    INT  NbTest,
                    const ElSTDNS list<Pt3dr> & PR3 ,
                    const ElSTDNS list<Pt2dr> & PF2 ,
                    REAL * Res_Dmin,
            bool  aModeRansac,
                    Pt3dr * aDirApprox
          )
{

     ELISE_ASSERT(PR3.size() == PF2.size(),"CombinatoireOFPA Dif Size");
     ELISE_ASSERT(INT(PR3.size())>=4,"CombinatoireOFPA, Size Insuffisant");

     ElRotation3D aRes(Pt3dr(0,0,0),0,0,0);
     * Res_Dmin = 1e8;

// #if (ELISE_unix || ELISE_MacOs || ELISE_MinGW)
     std::vector < Pt3dr > V3( PR3.begin() , PR3.end() );
     std::vector < Pt2dr > V2( PF2.begin() , PF2.end() );
// #else
//      ELISE_ASSERT(false,"No Vector interval init, with Visual");
//      std::vector < Pt3dr > V3;
//      std::vector < Pt2dr > V2;
// #endif
     std::list<Pt3dr>   L3(PR3);
     std::list<Pt2dr>   L2(PF2);


     INT aNB = ElMin((INT) V3.size(),ElMax(3,NbTest));

     int aNbTestMade= 0;
     for (INT k0= 0 ; k0<aNB ; k0++)
     {
          for (INT k1= k0+1 ; k1<aNB ; k1++)
          {
               for (INT k2= k1+1 ; k2<aNB ; k2++)
               {

                if (aModeRansac)
                        RansacTriplet(k0, k1, k2, (int)V3.size());
                L3.push_front(V3[k0]);
                L2.push_front(V2[k0]);
                L3.push_front(V3[k1]);
                L2.push_front(V2[k1]);
                L3.push_front(V3[k2]);
                L2.push_front(V2[k2]);

                    double aDist2 = ElMin3(dist8(V2[k0]-V2[k1]),dist8(V2[k1]-V2[k2]),dist8(V2[k2]-V2[k0]));

                    if (0)
                    {
                        std::cout << "AVANT OrientFromPtsAppu " << aNbTestMade  << " D=" << aDist2  << " NBt" << aNbTestMade  << " Nb=" << aNB << " " << V3.size() << "\n";
                        std::cout << V3[k0] << V3[k1] << V3[k2] << "\n";
                        std::cout << V2[k0] << V2[k1] << V2[k2] << "\n";
                    }

                    if (aDist2>1e-6)
                    {

                 REAL aDist;
                 INT  aNbSol;
                         ElRotation3D   aRot = OrientFromPtsAppui(TousDevant,L3,L2,&aDist,&aNbSol);


if (0)
{
    std::cout << "OrientFromPtsAppui DIST " << aDist   << " NbS " << aNbSol << "\n";
}

                 if ((aNbSol)&& (aDist < *Res_Dmin))
                 {
                            bool Ok = true;
                            if (aDirApprox !=0)
                            {
                                Ok = (scal(*aDirApprox,aRot.IRecVect(Pt3dr(0,0,1))) > 0);
                            }
                            if (Ok)
                            {
                       *Res_Dmin = aDist;
                       aRes = aRot;
                            }
                 }
                 aNbTestMade ++;
            }
            if (aModeRansac)
            {
               if (aNbTestMade==NbTest)
               {
                           return aRes;
               }
               else
               {
                   k0 = k1 = k2 = 0;
               }
            }
                L3.pop_front();
                L2.pop_front();
                L3.pop_front();
                L2.pop_front();
                L3.pop_front();
                L2.pop_front();
               }

          }
     }
     return aRes;
}



ElRotation3D  CamStenope::CombinatoireOFPA
              (
                    bool TousDevant,
                    INT  NbTest,
                    const ElSTDNS list<Pt3dr> & PR3 ,
                    const ElSTDNS list<Pt2dr> & PF2 ,
                    REAL * Res_Dmin,
                    Pt3dr * aDirApprox
          )
{
   return CombinatoireOFPAGen(TousDevant,NbTest,PR3,PF2,Res_Dmin,false,aDirApprox);
}

ElRotation3D  CamStenope::CombinatoireOFPA
              (
                    bool TousDevant,
                    INT  NbTest,
                    const ElSTDNS list<Appar23> & P23 ,
                    REAL * Res_Dmin,
                    Pt3dr * aDirApprox
          )
{
    ElSTDNS list<Pt3dr> PR3;
    ElSTDNS list<Pt2dr> PF2;

    ToSepList(PR3,PF2,P23);
    return CombinatoireOFPA(TousDevant,NbTest,PR3,PF2,Res_Dmin,aDirApprox);
}

ElRotation3D  CamStenope::RansacOFPA
              (
                    bool TousDevant,
                    INT  NbTest,
                    const ElSTDNS list<Appar23> & P23 ,
                    REAL * Res_Dmin,
                    Pt3dr * aDirApprox
          )
{
    ElSTDNS list<Pt3dr> PR3;
    ElSTDNS list<Pt2dr> PF2;

    ToSepList(PR3,PF2,P23);
    ElRotation3D aR =  CombinatoireOFPAGen(TousDevant,NbTest,PR3,PF2,Res_Dmin,true,aDirApprox);

    return  aR;
}






ElRotation3D  CamStenope::OrientFromPtsAppui
              (
                    bool TousDevant,
                    const ElSTDNS list<Appar23> & P23 ,
                    REAL * Res_Dmin,
                    INT   * NbSol
              )
{
    ElSTDNS list<Pt3dr> PR3;
    ElSTDNS list<Pt2dr> PF2;

    ToSepList(PR3,PF2,P23);
    return OrientFromPtsAppui(TousDevant,PR3,PF2,Res_Dmin,NbSol);
}


/*
REAL CamStenope::EcartProj(Pt2dr aPC2A,const ElCamera & CamB,Pt2dr aPC2B)
{
    Pt3dr RayB0,RayB1;
    CamB.F2toRayonR3(aPC2B,RayB0,RayB1);

    Pt3dr DirR = RayB1-RayB0;

    RayB0 = RayB0 + DirR * 10;
    RayB1 = RayB1 + DirR * 30;

    if (ElAbs(R3toL3(RayB0).z) < 1.0)
       RayB0 = RayB0 + DirR * 10;
    if (ElAbs(R3toL3(RayB1).z) < 1.0)
       RayB1 = RayB1 + DirR * 10;

    SegComp aSegCam(R3toC2(RayB0),R3toC2(RayB1));

    return aSegCam.dist(SegComp::droite,F2toC2(aPC2A));

}
*/


cParamIntrinsequeFormel * CamStenope::AllocParamInc(bool,cSetEqFormelles &)
{
   ELISE_ASSERT(false,"No def of CamStenope::AllocParamInc");
   return 0;
}




/*
ElRotation3D CamStenope::GPS_Orientation_From_Appuis
             (
                   const Appar23 & anAppar1,
                   const Appar23 & anAppar2
             )
{
     Pt3dr  aD1 = F2toRayonR3(anAppar1.pim);
     Pt3dr  aD2 = F2toRayonR3(anAppar2.pim);
}
*/

void  CamStenope::Set_GPS_Orientation_From_Appuis
                  (
                     const Pt3dr & aGPS,
                     const std::vector<Appar23> & aVApp,
                     int  aNbRansac
                  )
{
     int aNbPts = (int)aVApp.size();
     ELISE_ASSERT
     (
        aNbPts>=2,
        "CamStenope::GPS_Orientation_From_Appuis"
     );

     double anEcartMin = 1e15;
     ElRotation3D aSol(Pt3dr(0,0,0),0,0,0);

     while (1)
     {
         int aK1 = NRrandom3(aNbPts);
         int aK2 = NRrandom3(aNbPts);
         if (aK1!=aK2)
         {
            ElRotation3D aRC2M_Id(aGPS,0,0,0);
            SetOrientation(aRC2M_Id.inv());

            Appar23 anAp1= aVApp[aK1];
            Appar23 anAp2= aVApp[aK2];

            Pt3dr  aD1Cam = vunit(F2toDirRayonL3(anAp1.pim));
            Pt3dr  aD2Cam = vunit(F2toDirRayonL3(anAp2.pim));

            Pt3dr aD1Monde = vunit(anAp1.pter-aGPS);
            Pt3dr aD2Monde = vunit(anAp2.pter-aGPS);

            ElMatrix<REAL> aMatC2M = ComplemRotation(aD1Cam,aD2Cam,aD1Monde,aD2Monde);

            ElRotation3D aRC2M_Test(aGPS,aMatC2M,true);
            SetOrientation(aRC2M_Test.inv());

            double anEcart = SomEcartAngulaire(aVApp);

            if (anEcart < anEcartMin)
            {
                anEcartMin = anEcart;
                aSol = aRC2M_Test;
            }
         }
     }
     SetOrientation(aSol.inv());
}

void  CamStenope::ExpImp2Bundle(const Pt2di aGridSz, const std::string aName) const
{
    cXml_ScanLineSensor aSLS;

    double aZ = GetAltiSol();
    if (! ALTISOL_IS_DEF(aZ))
        aSLS.P1P2IsAltitude() = false;
    else
	aSLS.P1P2IsAltitude() = true;

    aSLS.LineImIsScanLine() = false;
    aSLS.GroundSystemIsEuclid() = true;
    aSLS.ImSz() = SzBasicCapt3D();

    aSLS.GridSz() = aGridSz;

    aSLS.StepGrid() = Pt2dr( double(SzBasicCapt3D().x)/aGridSz.x ,
		             double(SzBasicCapt3D().y)/aGridSz.y );

    //
    int aGr=0, aGc=0;
    for( aGr=0; aGr<aGridSz.x; aGr++ )
    {
        cXml_OneLineSLS aOL;
	aOL.IndLine() = aGr;
	for( aGc=0; aGc<aGridSz.y; aGc++ )
	{
	    cXml_SLSRay aOR;
	    aOR.IndCol() = aGc;

	    aOR.P1() = ImEtZ2Terrain(Pt2dr(aGr, aGc) , aZ+100);
            aOR.P2() = ImEtZ2Terrain(Pt2dr(aGr, aGc) , aZ);

	    aOL.Rays().push_back(aOR);
	}
	aSLS.Lines().push_back(aOL);
    }

    //export to XML format
    std::string aXMLFiTmp = "Bundle_" + aName;//see in what form the txt would be provided
    MakeFileXML(aSLS, aXMLFiTmp);
}




/*************************************************/
/*                                               */
/*    CamStenopeIdeale                           */
/*                                               */
/*************************************************/

CamStenopeIdeale::CamStenopeIdeale(bool isDistC2M,REAL Focale,Pt2dr centre,const std::vector<double> & ParamAF) :
    CamStenope(isDistC2M,Focale,centre,ParamAF)
{
}

ElDistortion22_Gen & CamStenopeIdeale::Dist()
{
   return ElDistortion22_Triviale::TheOne;
}

const ElDistortion22_Gen & CamStenopeIdeale::Dist() const
{
   return ElDistortion22_Triviale::TheOne;
}

CamStenopeIdeale::CamStenopeIdeale
(
   const CamStenopeIdeale & cam,
   const ElRotation3D & ORIENT
) :
    CamStenope
    (
        DistIsC2M(),
        cam._PrSten.focale(),
        cam._PrSten.centre(),
        cam.ParamAF()
    )
{
    SetOrientation(ORIENT);
}


CamStenopeIdeale  CamStenopeIdeale::CameraId(bool isDistC2M,const ElRotation3D & anOr)
{
    std::vector<double> NoAF;
    CamStenopeIdeale aCam(isDistC2M,1,Pt2dr(0,0),NoAF);
    aCam.SetOrientation(anOr);
    return aCam;
}


CamStenopeIdeale::CamStenopeIdeale ( const CamStenopeIdeale & cam) :
    CamStenope
    (
        DistIsC2M(),
        cam._PrSten.focale(),
        cam._PrSten.centre(),
        cam.ParamAF()
    )
{
    SetOrientation(cam._orient);
}


/*************************************************/
/*                                               */
/*    cCamStenopeGen                             */
/*                                               */
/*************************************************/

cCamStenopeGen::cCamStenopeGen
(
     CamStenope & aCS
) :
    CamStenope(aCS.DistIsC2M(),aCS.Focale(),aCS.PP(),aCS.ParamAF())
{
    mDist = &(aCS.Dist());

    HeritComplAndSz(aCS);
/*
    AddDistCompl(aCS.DistComplIsDir(),aCS.DistCompl());
    SetSz(aCS.Sz());
    if (aCS.HasRayonUtile())
    {
        SetRayonUtile(aCS.RayonUtile(),30);
    }
*/
}



/*************************************************/
/*                                               */
/*    cCamStenopeDistRadPol                      */
/*                                               */
/*************************************************/

cCamStenopeDistRadPol::cCamStenopeDistRadPol
    (  bool isDistC2M,
       REAL Focale,
       Pt2dr Centre,
       ElDistRadiale_PolynImpair aDist,
       const std::vector<double> & ParamAF,
       ElDistRadiale_PolynImpair * aRefDist,
       const Pt2di  & aSz
    ) :
    CamStenope(isDistC2M,Focale,Centre,ParamAF),
    mDist ( aRefDist ? *aRefDist : mDistInterne),
    mDistInterne (aDist)
{
   if (aSz!=ElCamera::TheSzUndef)
      SetSz(aSz);
}

const cCamStenopeDistRadPol * cCamStenopeDistRadPol::Debug_CSDRP() const
{
   return this;
}

void cCamStenopeDistRadPol::write(class  ELISE_fp & aFile)
{
        ELISE_ASSERT(ParamAF().size()==0,"cCamStenopeDistRadPol::write no AF");
    aFile.write(Focale());
    aFile.write(PP());
    aFile.write(DistIsC2M());
    mDist.write(aFile);
    aFile.write(Orient());
}
void cCamStenopeDistRadPol::write(const std::string & aName)
{
    ELISE_fp aFile(aName.c_str(),ELISE_fp::WRITE);
    write(aFile);
    aFile.close();
}

cCamStenopeDistRadPol * cCamStenopeDistRadPol::read_new(ELISE_fp & aFile)
{
       std::vector<double> NoAF;
    REAL aFoc = aFile.read((REAL  *)0);
    Pt2dr aPP = aFile.read((Pt2dr *)0);
    bool isDC2M = aFile.read((bool *)0);
    ElDistRadiale_PolynImpair aDist = ElDistRadiale_PolynImpair::read(aFile);

    cCamStenopeDistRadPol * aCam = new cCamStenopeDistRadPol (isDC2M,aFoc,aPP,aDist,NoAF);
    ElRotation3D aRot = aFile.read((ElRotation3D *)0);
        aCam->SetOrientation(aRot);


    return aCam;
}

cCamStenopeDistRadPol * cCamStenopeDistRadPol::read_new(const std::string & aName)
{
    ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);
    cCamStenopeDistRadPol *  aRes = read_new(aFile);
    aFile.close();
    return aRes;
}

const ElDistRadiale_PolynImpair & cCamStenopeDistRadPol::DRad() const
{
   return mDist;
}
ElDistRadiale_PolynImpair & cCamStenopeDistRadPol::DRad()
{
   return mDist;
}


const ElDistortion22_Gen & cCamStenopeDistRadPol::Dist() const
{
// std::cout << "KKKKKKKKKKKKKkkkk\n";
   return mDist;
}
ElDistortion22_Gen & cCamStenopeDistRadPol::Dist()
{
   return mDist;
}

cParamIntrinsequeFormel * cCamStenopeDistRadPol::AllocParamInc(bool isDC2M,cSetEqFormelles & aSet)
{
   return AllocDRadInc(isDC2M,aSet);
}

cParamIFDistRadiale * cCamStenopeDistRadPol::AllocDRadInc(bool isDC2M,cSetEqFormelles & aSetEq)
{



   cParamIFDistRadiale * aRes = aSetEq.NewIntrDistRad(isDC2M,this,5);
   aRes->SetFocFree(true);
   aRes->SetLibertePPAndCDist(true,true);

   return aRes;
}



/*************************************************/
/*                                               */
/*          cCamStenopeModStdPhpgr               */
/*                                               */
/*************************************************/

cCamStenopeModStdPhpgr::cCamStenopeModStdPhpgr
(
     bool isDistC2M,
     REAL aFocale,
     Pt2dr aCentre,
     cDistModStdPhpgr aDist,
     const std::vector<double> & ParamAF
)  :
   cCamStenopeDistRadPol(isDistC2M,aFocale,aCentre,aDist,ParamAF,&mDist),
   mDist (aDist)
{
/*
   if (DistC2M)
      SetDistInverse();
   else
      SetDistDirecte();
*/
}

cDistModStdPhpgr & cCamStenopeModStdPhpgr::DModPhgrStd()
{
    return mDist;
}

const cDistModStdPhpgr & cCamStenopeModStdPhpgr::DModPhgrStd() const
{
    return mDist;
}

ElDistortion22_Gen & cCamStenopeModStdPhpgr::Dist()
{
   return mDist;
}
const ElDistortion22_Gen & cCamStenopeModStdPhpgr::Dist()  const
{
   return mDist;
}

cParamIntrinsequeFormel * cCamStenopeModStdPhpgr::AllocParamInc(bool isDC2M,cSetEqFormelles & aSet)
{
   return AllocPhgrStdInc(isDC2M,aSet);
}

cParamIFDistStdPhgr * cCamStenopeModStdPhpgr::AllocPhgrStdInc(bool isDC2M,cSetEqFormelles & aSetEq)
{

   cParamIFDistStdPhgr * aRes = aSetEq.NewIntrDistStdPhgr (isDC2M,this,5);
   aRes->SetFocFree(true);
   aRes->SetLibertePPAndCDist(true,true);
   aRes->SetParam_Aff_Free();
   aRes->SetParam_Dec_Free();

   return aRes;
}


/*************************************************/
/*                                               */
/*    cCamStenopeDistPolyn                       */
/*                                               */
/*************************************************/

cCamStenopeDistPolyn::cCamStenopeDistPolyn
(
     bool isDC2M,
     REAL Focale,
     Pt2dr Centre,
     const ElDistortionPolynomiale & aDist,
     const std::vector<double> &     aParamAF
)   :
    CamStenope (isDC2M,Focale,Centre,aParamAF),
    mDist      (aDist)
{
}

ElDistortion22_Gen & cCamStenopeDistPolyn::Dist()
{
   return mDist;
}
const ElDistortion22_Gen & cCamStenopeDistPolyn::Dist()  const
{
   return mDist;
}

const ElDistortionPolynomiale & cCamStenopeDistPolyn::DistPol() const
{
   return mDist;
}

/*************************************************/
/*                                               */
/*    cCamStenopeDistHomogr                      */
/*                                               */
/*************************************************/

cCamStenopeDistHomogr::cCamStenopeDistHomogr
(
     bool isDC2M,
     REAL Focale,
     Pt2dr Centre,
     const cDistHomographie & aDist,
     const std::vector<double> &  aParamAF
)  :
    CamStenope (isDC2M,Focale,Centre,aParamAF),
    mDist      (aDist)
{
}

ElDistortion22_Gen & cCamStenopeDistHomogr::Dist()
{
   return mDist;
}
const ElDistortion22_Gen & cCamStenopeDistHomogr::Dist()  const
{
   return mDist;
}


const  cElHomographie & cCamStenopeDistHomogr::Hom() const
{
   return mDist.Hom();
}



/*************************************************/
/*                                               */
/*       cAnalyseZoneLiaison                     */
/*                                               */
/*************************************************/

cAnalyseZoneLiaison::cAnalyseZoneLiaison()
{
}

void cAnalyseZoneLiaison::Reset()
{
   mMat = RMat_Inertie();
   mVPts.clear();
}

const std::vector<Pt2dr> & cAnalyseZoneLiaison::VPts() const
{
   return mVPts;
}


void cAnalyseZoneLiaison::AddPt(const Pt2dr & aPt)
{
   mVPts.push_back(aPt);
   mMat.add_pt_en_place(aPt.x,aPt.y);
}

double   cAnalyseZoneLiaison::Score(double ExposantDist,double ExposantPds)
{
   if ( mVPts.size() <= 1)
      return -1;

   bool Ok=false;
   for (int aK=1 ; (!Ok) && (aK<int(mVPts.size()))  ; aK++)
   {
        if (mVPts[aK-1] != mVPts[aK])
           Ok=true;
   }

   if (! Ok)
      return -1;
 // std::cout << "cAnalyseZoneLiaison::Score  " << mVPts.size() << "\n";
    SegComp aSeg = seg_mean_square(mMat,1.0);
 // std::cout << "hkjhkjhfdd--------------- \n";

    double aRes = 0.0;
    double aSPds = 0;

    for (int aK=0 ; aK<int(mVPts.size()) ; aK++)
    {
        double aD = ElAbs(aSeg.ordonnee(mVPts[aK]));
        aRes += pow(aD,ExposantDist);
        aSPds++;
    }
    aRes /= aSPds;
    aRes = pow(aRes,1/ExposantDist);

    return aRes * pow(aSPds,ExposantPds);

}

/*************************************************/
/*                                               */
/*    cCamStenopeDistRadPol                      */
/*    cCamStenopeGrid                            */
/*                                               */
/*************************************************/

cCamStenopeGrid::cCamStenopeGrid
    (
        const double & aFoc,
        const Pt2dr & aCentre,
        cDistCamStenopeGrid * aDCSG,
        const Pt2di  & aSz,
         const std::vector<double> & ParamAF
    ) :
    CamStenope(false,aFoc,aCentre,ParamAF),
    mDGrid    (aDCSG)
{
   mDist = aDCSG;
   if (aSz!=ElCamera::TheSzUndef)
      SetSz(aSz);
}

cCamStenopeGrid * cCamStenopeGrid::Alloc
                  (
                       double aRayInv,
                       const CamStenope & aCS,
                       Pt2dr aStepGr,
                       bool  doDir,
                       bool  doInv
                  )
{


  double aSauvSc = aCS.ScaleCamNorm();
  Pt2dr  aSauvTr = aCS.TrCamNorm();
  const_cast<CamStenope &>(aCS).UnNormalize();

 cDistCamStenopeGrid * aDist =
       cDistCamStenopeGrid::Alloc(!aCS.DistIsDirecte(),aRayInv,aCS,aStepGr,doDir,doInv);

  cCamStenopeGrid * aRes =  new cCamStenopeGrid(aCS.Focale(),aCS.PP(),aDist,aCS.Sz(),aCS.ParamAF());

  aRes->mSzPixel =  aCS.SzPixelBasik();

  // HERE

  aRes->CamHeritGen(aCS,false,false);

  const_cast<CamStenope &>(aCS).StdNormalise(true,true,aSauvSc,aSauvTr);




  return aRes;
}

bool cCamStenopeGrid::IsGrid() const
{
    return true;
}

ElDistortion22_Gen  * cCamStenopeGrid::DistPreCond() const
{
   return mDGrid->mPreC;
}


Pt2dr cCamStenopeGrid::L2toF2AndDer(Pt2dr aP,Pt2dr & aGX,Pt2dr & aGY)
{
    double aF = Focale();
    aP = aP * aF + PP();
    aP = mDGrid->DirectAndDer(aP,aGX,aGY);

    aGX = aGX * aF;
    aGY = aGY * aF;

    return aP;


}



CamStenope * CamCompatible_doublegrid(const std::string & aNameFile)
{

    cElXMLTree aTree(aNameFile);

   if (aTree.Get("doublegrid") && aTree.Get("focal") &&  aTree.Get("usefull-frame"))
   {
      std::vector<double> NoPAF;
      std::string aDir,aFile;
      SplitDirAndFile(aDir,aFile,aNameFile);

      cDbleGrid::cXMLMode aXmlMode(true);  // 2 Swap
      cDbleGrid * aGrid = new  cDbleGrid(aXmlMode,aDir,aFile);

      cDistCamStenopeGrid * aDCG = new cDistCamStenopeGrid(0,aGrid);  // 0 : Pas de Preconditionneur

      cElXMLTree * aTrRect = aTree.Get("usefull-frame");
      Pt2di aSz (aTrRect->GetUniqueValInt("w"),aTrRect->GetUniqueValInt("h"));

      return new cCamStenopeGrid
                 (
                      1.0,
                      Pt2dr(0,0),
                      aDCG,
                      aSz,
                      NoPAF
                 );
   }

   return Std_Cal_From_File(aNameFile);

}
/*
*/


/************************************************************************/
/*                                                                      */
/*          Camera Ortho - Proj Ident                                   */
/*                                                                      */
/************************************************************************/

           //    ElProjIdentite


Pt2dr ElProjIdentite::Proj(Pt3dr aP) const
{
    return Pt2dr (aP.x,aP.y);
}

Pt3dr ElProjIdentite::DirRayon(Pt2dr) const
{
    return Pt3dr(0,0,1);
}

void   ElProjIdentite::Diff(ElMatrix<REAL> &,Pt3dr) const
{
   ELISE_ASSERT(false,"No ElProjIdentite::Diff");
}

void  ElProjIdentite::Rayon(Pt2dr aP,Pt3dr &p0,Pt3dr & p1) const
{
      p0 = Pt3dr(aP.x,aP.y,0);
      p1 = Pt3dr(aP.x,aP.y,1);
}

ElProjIdentite ElProjIdentite::TheOne;

           //  cCameraOrtho

bool    cCameraOrtho::HasOpticalCenterOfPixel() const
{
   return false;
}


cCameraOrtho::cCameraOrtho(const Pt2di & aSz) :
   ElCamera(false,eProjectionOrtho)
{
   ElCamera::SetSz(aSz);
   SetOrientation(ElRotation3D(Pt3dr(0,0,1),0,0,0));
}


Pt3dr cCameraOrtho::R3toL3(Pt3dr aP) const
{
   aP =  _orient.ImVect(aP);
   const Pt3dr & aTr = _orient.tr();

   return Pt3dr
          (
              aP.x *aTr.z +  aTr.x,
              aP.y *aTr.z +  aTr.y,
              aP.z
          );
}

double cCameraOrtho::ResolutionSol() const
{
    return  _orient.tr().z;
}
double cCameraOrtho::ResolutionSol(const Pt3dr &) const
{
    return  ResolutionSol();
}

Pt3dr cCameraOrtho::L3toR3(Pt3dr aP) const
{
   const Pt3dr & aTr = _orient.tr();

   aP =   Pt3dr
          (
              (aP.x -  aTr.x) / aTr.z,
              (aP.y -  aTr.y) / aTr.z,
              aP.z
          );

   return   _orient.IRecVect(aP);
}


ElProj32   &  cCameraOrtho::Proj()
{
   return ElProjIdentite::TheOne;
}

const ElProj32   &  cCameraOrtho::Proj() const
{
   return ElProjIdentite::TheOne;
}

ElDistortion22_Gen & cCameraOrtho::Dist()
{
   return ElDistortion22_Triviale::TheOne;
}


const ElDistortion22_Gen & cCameraOrtho::Dist() const
{
   return ElDistortion22_Triviale::TheOne;
}

void cCameraOrtho::InstanceModifParam(cCalibrationInternConique & aParam)  const
{
    aParam.PP() = Pt2dr(12345678,87654321);
    aParam.F()  = 0;
}

cCameraOrtho * cCameraOrtho::Alloc(const Pt2di & aSz)
{
    cCameraOrtho * aRes = new cCameraOrtho(aSz);
    ElSeg3D  aSeg = aRes->F2toRayonR3(aSz/2.0);

    aRes->mCentre = aSeg.ProjOrtho(Pt3dr(0,0,0));


    return aRes;

}

Pt3dr cCameraOrtho::ImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
     return mCentre + F2toDirRayonR3(aP) * aZ;
}
Pt3dr cCameraOrtho::NoDistImEtProf2Terrain(const Pt2dr & aP,double aZ) const
{
     return ImEtProf2Terrain(aP,aZ);
}


Pt3dr  cCameraOrtho::OrigineProf() const
{
    return mCentre;
}

bool  cCameraOrtho::HasOrigineProf() const
{
    // Modif MPD le 17/06/2014 ; semle + logique comme cela et est utilise dans ElCamera::PIsVisibleInImage
    return false;
}


double cCameraOrtho::SzDiffFinie() const
{
    return 1.0;
}


/*
class cCameraOrtho : public ElCamera
{
    public :
       cCameraOrtho();
    private :
         Pt3dr L3toR3(Pt3dr) const;

         ElDistortion22_Gen   &  Dist()        ;
         const ElDistortion22_Gen   &  Dist() const  ;
         ElProj32 &        Proj()       ;
         const ElProj32       &  Proj() const ;
};
*/

#if (0)
#endif




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
