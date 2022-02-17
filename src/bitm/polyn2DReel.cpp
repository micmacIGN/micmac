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

/*********************************************************/
/*                                                       */
/*                Monome2dReal                           */
/*                                                       */
/*********************************************************/

REAL Monome2dReal::operator () (Pt2dr aP) const
{
     return pow(aP.x/mAmpl,mD0X) *  pow(aP.y/mAmpl,mD0Y);
}

Fonc_Num Monome2dReal::FNum() const
{
     return PowI(FX/mAmpl,mD0X) *  PowI(FY/mAmpl,mD0Y);
}

void Monome2dReal::Show(bool X) const
{
    INT degre = (X?DegreX():DegreY());
    if (degre)
    {
        std::cout << (X ? "X" : "Y");
        if (degre>1) std::cout << degre;
    }
}


Pt2dr Monome2dReal::grad(Pt2dr aP) const
{
     REAL dX = mD0X                                                             ?
               ((mD0X / mAmpl) * pow(aP.x/mAmpl,mD0X-1) * pow(aP.y/mAmpl,mD0Y)) :
               0                                                                ;
     REAL dY = mD0Y                                                             ?
               ((mD0Y / mAmpl) * pow(aP.x/mAmpl,mD0X) * pow(aP.y/mAmpl,mD0Y-1)) :
               0                                                                ;

     return   Pt2dr (dX,dY);
}

void Monome2dReal::SetAmpl(REAL anAmpl)
{
    mAmpl = anAmpl;
}


INT Monome2dReal::DegreTot() const
{
    return mD0X + mD0Y;
}

REAL Monome2dReal::CoeffMulNewAmpl (REAL NewAmpl) const
{
   return pow(NewAmpl/mAmpl,DegreTot());
}


/*********************************************************/
/*                                                       */
/*                Polynome2dReal                         */
/*                                                       */
/*********************************************************/

#define   P2DDMax 100


Polynome2dReal::Polynome2dReal(INT aDMax,REAL anAmpl) :
        mAmpl (anAmpl),
        mDMax (aDMax)
{
    BENCH_ASSERT(mDMax<P2DDMax);

    for (INT aDTot = 0 ; aDTot<= aDMax ;  aDTot++)
    {
         for (INT aDx =0 ; aDx<= aDTot ; aDx++)
         {
             mMons.push_back(Monome2dReal(aDx,aDTot-aDx,anAmpl));
             mCoeff.push_back(1.0);
         }
    }
}

INT Polynome2dReal::DMax() const
{
   return mDMax;
}

REAL Polynome2dReal::CoeffNewAmpl (INT k,REAL NewAmpl) const
{
   return (k<(INT) mCoeff.size()) ?
          mCoeff[k]*mMons[k].CoeffMulNewAmpl(NewAmpl)  :
          0;
}

Polynome2dReal::Polynome2dReal
(
         const Polynome2dReal & aPol1,
         REAL                   aCoef1,
         const Polynome2dReal & aPol2,
         REAL                   aCoef2
)   :
        mAmpl (ElMax(aPol1.mAmpl,aPol2.mAmpl)),
        mDMax (ElMax(aPol1.mDMax,aPol2.mDMax))

{
    INT aK =0;
    for (INT aDTot = 0 ; aDTot<= mDMax ;  aDTot++)
    {
         for (INT aDx =0 ; aDx<= aDTot ; aDx++)
         {
             mMons.push_back(Monome2dReal(aDx,aDTot-aDx,mAmpl));
             mCoeff.push_back
             (
                  aCoef1 * aPol1.CoeffNewAmpl(aK,mAmpl)
                + aCoef2 * aPol2.CoeffNewAmpl(aK,mAmpl)
             );
             aK++;
         }
    }
}

Polynome2dReal Polynome2dReal::operator + (const Polynome2dReal & other) const
{
    return Polynome2dReal (*this,1.0,other,1.0);
}

Polynome2dReal Polynome2dReal::operator - (const Polynome2dReal & other) const
{
    return Polynome2dReal (*this,1.0,other,-1.0);
}

Polynome2dReal Polynome2dReal::operator * (REAL aVal) const
{
    return Polynome2dReal (*this,aVal,*this,0.0);
}

Polynome2dReal Polynome2dReal::operator / (REAL aVal) const
{
    return Polynome2dReal (*this,1.0/aVal,*this,0.0);
}



Polynome2dReal Polynome2dReal::PolyDegre1(REAL aV0,REAL aVX, REAL aVY)
{
    Polynome2dReal aRes(1,1.0);
    aRes.SetDegre1(aV0,aVX,aVY,true);
    return aRes;
}
/*
*/




void Polynome2dReal::SetDegre1(REAL aV0,REAL aVX, REAL aVY,bool AnnulOthers)
{
    for (INT k=0 ; k<(INT)mCoeff.size() ; k++)
    {
        if (mMons[k].DegreTot() >1)
        {
             if (AnnulOthers) 
                 mCoeff[k] =0;
        }
        else if (mMons[k].DegreTot() == 0)
        {
              mCoeff[k] = aV0 ;
        }
        else if (mMons[k].DegreX() == 1)
        {
              mCoeff[k] = aVX * mAmpl;
        }
        else
        {
              mCoeff[k] = aVY * mAmpl;
        }
    }
}



static REAL P2DPowX[P2DDMax+1];
static REAL P2DPowY[P2DDMax+1];

static void PrepP2D(Pt2dr aP,REAL anAmpl,INT aDeg)
{
   aP = aP/anAmpl;
   P2DPowX[0] = 1.0;
   P2DPowY[0] = 1.0;

   for (INT k=1 ; k<= aDeg ; k++)
   {
       P2DPowX[k] = aP.x * P2DPowX[k-1];
       P2DPowY[k] = aP.y * P2DPowY[k-1];
   }
}


REAL Polynome2dReal::operator () (Pt2dr aP) const
{
   PrepP2D(aP,mAmpl,mDMax);
   REAL aRes = 0.0;
   for (INT k=0; k<(INT)mMons.size() ; k++)
       aRes += mCoeff[k] * P2DPowX[mMons[k].DegreX()] * P2DPowY[mMons[k].DegreY()];

   return aRes;



/*
   REAL aRes = 0.0;
   for (INT k=0; k<(INT)mMons.size() ; k++)
       aRes += mCoeff[k] * mMons[k](aP);

   return aRes;
*/
}

Pt2dr Polynome2dReal::grad(Pt2dr aP) const
{
   Pt2dr aRes (0,0);
   for (INT k=0; k<(INT)mMons.size() ; k++)
   {
       aRes +=  mMons[k].grad(aP) * mCoeff[k];
   }

   return aRes;
    
}


Fonc_Num  Polynome2dReal::FNum() const
{
   Fonc_Num aRes = 0.0;
   for (INT k=0; k<(INT)mMons.size() ; k++)
       aRes = aRes + mCoeff[k] * mMons[k].FNum();

   return aRes;
}


INT Polynome2dReal::NbMonome(void)  const
{
    return (INT) mMons.size();
}

const Monome2dReal & Polynome2dReal::KthMonome(int aK) const
{
   AssertIndexeValide(aK);
   return mMons[aK];
}

void  Polynome2dReal::SetCoeff(int aK,REAL  aCoeff)
{
   AssertIndexeValide(aK);
    mCoeff[aK] = aCoeff;
}

INT Polynome2dReal::DegreX(INT aK) const
{
   AssertIndexeValide(aK);
   return mMons[aK].DegreX();
}

INT Polynome2dReal::DegreY(INT aK) const
{
   AssertIndexeValide(aK);
   return mMons[aK].DegreY();
}

INT Polynome2dReal::DegreTot(INT aK) const
{
   AssertIndexeValide(aK);
   return mMons[aK].DegreTot();
}






REAL  Polynome2dReal::Coeff(int aK) const
{
    return mCoeff[aK];
}
REAL &  Polynome2dReal::Coeff(int aK) 
{
    return mCoeff[aK];
}


void  Polynome2dReal::AssertIndexeValide(INT aK) const
{
    ELISE_ASSERT
    (
        (aK>=0) && (aK<(INT)mCoeff.size()),
        "Invalide Poly Indexe"
    );
}

REAL Polynome2dReal:: Ampl() const
{
   return mAmpl;
}

void Polynome2dReal::Show(INT aK) const
{
     std::cout << mCoeff[aK] * mMons[aK](Pt2dr(1,1));
     mMons[aK].Show(true);
     mMons[aK].Show(false);
}

void Polynome2dReal::Show() const
{
    for (INT aK=0 ; aK<(INT)mCoeff.size() ; aK++)
    {
        if (aK>=1)
           std::cout << "+";
        Show(aK);
    }
}


Polynome2dReal Polynome2dReal::MapingChScale(REAL aChSacle) const
{
    Polynome2dReal aRes(*this);
    aRes.mAmpl *= aChSacle;

    for (INT k=0 ; k<(INT)mCoeff.size() ; k++)
    {
        aRes.mMons[k].SetAmpl(aRes.mAmpl);
        aRes.mCoeff[k] *= aChSacle;
    }
    return aRes;
}


void Polynome2dReal:: write(ELISE_fp & aFile) const
{
     aFile.write(mCoeff);
     aFile.write(mAmpl);
     aFile.write(mDMax);
}

Polynome2dReal Polynome2dReal::read(ELISE_fp & aFile)
{
   std::vector<REAL> aCoeff = aFile.read(&aCoeff);
   REAL anAmpl =  aFile.read(&anAmpl);
   INT aDmax =  aFile.read(&aDmax);

   Polynome2dReal aRes(aDmax,anAmpl);

   for (int aK=0; aK<(INT)aCoeff.size() ; aK++)
       aRes.SetCoeff(aK,aCoeff[aK]);
   return aRes;
}

std::vector<double>  Polynome2dReal::ToVect() const
{
    return mCoeff;
}
Polynome2dReal Polynome2dReal::FromVect(const std::vector<double>& aCoef,double anAmpl)
{
    int aDeg = 0;
    while (   ((aDeg+2)*(aDeg+1))/2 < int(aCoef.size()))
        aDeg++;
    ELISE_ASSERT( ((aDeg+2)*(aDeg+1))/2 == int(aCoef.size()),"Polynome2dReal::FromVect");

    Polynome2dReal aRes(aDeg,anAmpl);
    aRes.mCoeff = aCoef;
    return aRes;
}

// Polynome2dReal 
// std::vector<double> ToVect() const;
// static Polynome2dReal FromVect(const std::vector<double>&);

Polynome2dReal LeasquarePol2DFit
               (
                    int                           aDegre,
                    const std::vector<Pt2dr> &    aVP,
                    const std::vector<double>     aVals,
                    const std::vector<double> *   aVPds
               )
{
    Polynome2dReal aRes(aDegre,1.0);
    int aNbP =aVP.size();
    int aNbM = aRes.NbMonome();
    ELISE_ASSERT(int(aVals.size())==aNbP,"Size inc in LeasquarePol2DFit");
    ELISE_ASSERT((aVPds==0) || (int(aVPds->size())==aNbP),"Size inc in LeasquarePol2DFit");

    L2SysSurResol aSys(aNbM);
    std::vector<double> aVCoef(aNbM,0.0);

    for (int aKpt=0 ; aKpt< aNbP ; aKpt++)
    {
        for (int aKMon=0 ; aKMon< aNbM ; aKMon++)
        {
              const Monome2dReal &  aMon = aRes.KthMonome(aKMon);
              aVCoef[aKMon] = aMon(aVP[aKpt]);
              double aPds = aVPds ?  (*aVPds)[aKMon] : 1.0;
              aSys.AddEquation(aPds,&(aVCoef[0]),aVals[aKpt]);
        }
    }
    bool Ok;
    Im1D_REAL8  aSol = aSys.GSSR_Solve(&Ok);
    double * aD= aSol.data();

    return Polynome2dReal::FromVect(std::vector<double>(aD,aD+aNbM),1.0);
}



Polynome2dReal LeasquarePol2DFit
               (
                    int                           aDegre,
                    const std::vector<Pt2dr> &    aVP,
                    const std::vector<double>     aVals,
                    const Polynome2dReal &        aLastPol,
                    double                        aPropEr,
                    double                        aFactEr,
                    double                        aErMin
               )
{
    int aNbPts = aVP.size();
    std::vector<double>  aVErr;
    for (int aKp=0 ; aKp<aNbPts ; aKp++)
    {
        Pt2dr aPt = aVP[aKp];
        double anEr = ElAbs(aVals[aKp]-aLastPol(aPt));
        aVErr.push_back(anEr);
    }
    std::vector<double> aVErSorted =  aVErr;
    double aErStd = KthValProp(aVErSorted,aPropEr);


    std::vector<double>  aVPds;
    for (int aKp=0 ; aKp<aNbPts ; aKp++)
    {
        double aPds = ((aVErr[aKp]+aErMin) / aErStd) * aFactEr;
        aPds = 1 / (1 + ElSquare(aPds));
        aVPds.push_back(aPds);
    }
    return LeasquarePol2DFit(aDegre,aVP,aVals,&aVPds);
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
