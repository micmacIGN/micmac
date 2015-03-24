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

#include "general/all.h"
#include "private/all.h"
#include "ReducHom.h"

using namespace NS_ReducHoms;

/*************************************************/
/*                                               */
/*                cMesPtIm                       */
/*                                               */
/*************************************************/

/*
cMesPtIm::cMesPtIm(const Pt2dr & aPt,cImagH * anIm) :
   mPt (aPt),
   mIm (anIm)
{
}

cImagH * cMesPtIm::Im() const
{
   return mIm;
}

const Pt2dr  & cMesPtIm::Pt() const
{
   return mPt;
}
*/


/*************************************************/
/*                                               */
/*                cPtHom                         */
/*                                               */
/*************************************************/


std::list<cPtHom *> cPtHom::mReserve;
std::list<cPtHom *> cPtHom::mAllExist;
cPtHom::cPtHom() 
{
   Clear();
}

void  cPtHom::Clear()
{
    mCptArc = 0;
    mMesures.clear();
    mCoherent = true;
}


void  cPtHom::Recycle()
{
    Clear();
    mReserve.push_back(this);
}


cPtHom * cPtHom::Alloc()
{
   if (mReserve.empty())
   {
      cPtHom * aRes =  new cPtHom;
      mAllExist.push_back(aRes);
      return aRes;
   }
   cPtHom * aRes = mReserve.back();
   mReserve.pop_back();
   return aRes;
}

void  cPtHom::IncrCptArc()
{
   mCptArc++;
}

std::vector<Pt2dr> VecOnePt(const Pt2dr & aP)
{
   std::vector<Pt2dr> aV;
   aV.push_back(aP);
   return aV;
}

void AddOnePtUnique(std::vector<Pt2dr> & aV,const Pt2dr & aP)
{
    for (int aK=0 ; aK<int(aV.size()) ; aK++)
    {
         if (aV[aK] == aP)
            return;
    }
    aV.push_back(aP);
}

bool cPtHom::OkAddI2(cImagH * aI2,const Pt2dr & aP2)
{
    IncrCptArc();
    aI2->SetPHom(aP2,this);
    std::vector<Pt2dr> & aV = mMesures[aI2];
    if (aV.size() !=0)  
        mCoherent = false;

    AddOnePtUnique(aV,aP2);
/*
    if (0)
    {
         
        std::cout << "INC IN PAIR " << aP2 << mMesures[aI2] <<  (mMesures[aI2]< aP2)  << ( aP2< mMesures[aI2])<<  "\n";
        std::cout << "NB MESURES " <<  mMesures.size() << " Nb A " << mCptArc << "\n";

        getchar();
    }
*/
    return false;
}

int cPtHom::NbIm() const
{
   return mMesures.size();
}


cPtHom * cPtHom::NewGerm(cImagH * aI1,const Pt2dr & aP1,cImagH* aI2,const Pt2dr & aP2)
{
   cPtHom * aRes = Alloc();
   aRes->mMesures[aI1] = VecOnePt(aP1);
   aRes->mMesures[aI2] = VecOnePt(aP2);
   aRes->IncrCptArc();

   return aRes;
}

bool cPtHom::OkAbsorb(cPtHom * aH2)
{
   if (! aH2->mCoherent) 
      mCoherent = false;

   for 
   (
        std::map<cImagH*,std::vector<Pt2dr> >::iterator itM2= aH2->mMesures.begin();
        itM2 != aH2->mMesures.end();
        itM2++
   )
   {
        cImagH * aI2 = itM2->first;
        const std::vector<Pt2dr> &  aVP2 = itM2->second;
        std::vector<Pt2dr> &   aVP1 = mMesures[aI2];

        if (aVP1.size() != 0)
        {
           mCoherent =false;
        }

        for (int aK2=0 ; aK2 <int(aVP2.size()) ; aK2++)
        {
            aI2->SetPHom(aVP2[aK2],this);
            AddOnePtUnique(aVP1,aVP2[aK2]);
        }
   }


   mCptArc += aH2->mCptArc;
   IncrCptArc();
   aH2->Recycle();

   return mCoherent;
}



class cStatH
{
    public :
        cStatH () :
           mNbPop  (0),
           mNbPopC (0),
           mNbA    (0)
        {
        }

        int mNbPop;
        int mNbPopC;
        int mNbA;
};


void cPtHom::ShowAll()
{
   std::map<int,cStatH> aDicoStat;
   for 
   ( 
         std::list<cPtHom  *>::iterator itH=mAllExist.begin();
         itH!=mAllExist.end();
         itH++
   )
   {
       const cPtHom & aH = **itH;  // Sinon recycle
       if (aH.mCptArc)
       {
           int aNbS = aH.mMesures.size();
           cStatH &  aStat = aDicoStat[aNbS];
           if (aNbS==1)
           {
               for 
               (
                  std::map<cImagH*,std::vector<Pt2dr> >::const_iterator itC=aH.mMesures.begin();
                  itC!=aH.mMesures.end();
                  itC++
               )
               {
                  std::cout << "CPLE " << itC->first->Name() << " " << itC->second.size()  << " NbA " << aH.mCptArc << "\n";
               }
           }
           aStat.mNbPop++;
           if (aH.mCoherent)
           {
              aStat.mNbPopC ++;
              aStat.mNbA += aH.mCptArc;
           }
       }
         
/*
        if ((aH.mMesures.size() != (aH.mCptArc+1)))
        {
           std::cout << " NbS " << aH.mMesures.size() << " NbA " << aH.mCptArc  << " CoH " << aH.mCoherent << "\n";
        }
*/
   }

   for (std::map<int,cStatH>::const_iterator itH=aDicoStat.begin() ; itH!=aDicoStat.end() ; itH++)
   {
       const cStatH & aStat = itH->second;
       int aMul= itH->first;
       int aPop =  aStat.mNbPop;
       int aPopC =  aStat.mNbPopC;
       std::cout << "Mul " << aMul  << " NbS " << aPop << " \% Coh " << ((aPopC*100.0)/aPop);

       if (aPopC)
       {
             std::cout << " Densite-Arc " << (aStat.mNbA/double(aPopC*aMul*(aMul-1))) ;
       }

       std::cout << "\n";
   }
   getchar();
}




/*
cMesPtIm * cPtHom::GetMesOfImSVP(cImagH * anIm)
{
   for 
   (
      std::list<cMesPtIm>::iterator itM=mMesures.begin();
      itM!=mMesures.end();
      itM++
   )
   {
       if (itM->Im()==anIm)
          return &(*itM);
   }
   return 0;
}

cMesPtIm &  cPtHom::GetMesOfIm(cImagH * anIm)
{
   cMesPtIm * aRes = GetMesOfImSVP(anIm);
   if (aRes==0)
   {
      std::cout << "For name " << anIm->Name();
      ELISE_ASSERT(false,"cPtHom::GetMesOfIm");
   }
   return *aRes;
}
*/


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
