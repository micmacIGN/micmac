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


#include "TiepTri.h"


   // =================== cResulMultiImRechCorrel ===== 
   // =================== cResulMultiImRechCorrel ===== 
   // =================== cResulMultiImRechCorrel ===== 

cResulMultiImRechCorrel::cResulMultiImRechCorrel(const cIntTieTriInterest & aPMaster) :
    mPMaster (aPMaster),
    mScore   (TT_MaxCorrel),
    mAllInit (true),
    mNbSel   (0)
{
}

double cResulMultiImRechCorrel::square_dist(const cResulMultiImRechCorrel & aR2) const
{
     return square_euclid(mPMaster.mPt,aR2.mPMaster.mPt);
}
void cResulMultiImRechCorrel::AddResul(const cResulRechCorrel aRRC,int aNumIm)
{
   if (aRRC.IsInit())
   {
       mScore = ElMin(mScore,aRRC.mCorrel);
       mVRRC.push_back(aRRC);
       mVIndex.push_back(aNumIm);
       mVSelec.push_back(true);
       mNbSel++;
   }
   else
   {
       mAllInit = false;
   }
}

bool cResulMultiImRechCorrel::AllInit() const  
{
   return mAllInit ;
}
bool cResulMultiImRechCorrel::IsInit() const  
{
   return mAllInit && (mVRRC.size() !=0) ;
}
double cResulMultiImRechCorrel::Score() const 
{
   return mScore;
}
const std::vector<cResulRechCorrel > & cResulMultiImRechCorrel::VRRC() const 
{
   return mVRRC;
}
std::vector<cResulRechCorrel > & cResulMultiImRechCorrel::VRRC() 
{
   return mVRRC;
}

const cIntTieTriInterest & cResulMultiImRechCorrel::PIMaster() const 
{
   return  mPMaster;
}
cIntTieTriInterest & cResulMultiImRechCorrel::PIMaster() 
{
   return  mPMaster;
}

Pt2di  cResulMultiImRechCorrel::PtMast() const 
{
   return PIMaster().mPt;
}

const std::vector<int> &   cResulMultiImRechCorrel::VIndex()   const 
{
   return  mVIndex;
}

int &       cResulMultiImRechCorrel::HeapIndexe ()       {return mHeapIndexe;}
const int & cResulMultiImRechCorrel::HeapIndexe () const {return mHeapIndexe;}

void cResulMultiImRechCorrel::CalculScoreAgreg(double Epsilon,double anExp,double aSign)
{
    mScore = 0.0;
    for (int aK=0 ; aK<int(mVSelec.size()) ; aK++)
    {
        if (mVSelec[aK])
           mScore += pow(1/(Epsilon + (1-mVRRC[aK].mCorrel)),anExp);
    }
    mScore *= aSign;
}

void  cResulMultiImRechCorrel::SetAllSel()
{
    for (int aK=0 ; aK<int(mVSelec.size()) ; aK++)
        SetSelec(aK,true);
}

void  cResulMultiImRechCorrel::SetSelec(int aK,bool aVal)
{
     if (mVSelec[aK] != aVal)
     {
          mVSelec[aK] = aVal;
          mNbSel += (aVal ? 1 : -1);
     }
}

int cResulMultiImRechCorrel::NbSel() const {return mNbSel;}

// std::vector<bool>  &         cResulMultiImRechCorrel::VSelec()       {return mVSelec; }
// const std::vector<bool>  &   cResulMultiImRechCorrel::VSelec() const {return mVSelec; }


void cResulMultiImRechCorrel::SuprUnSelect()
{
   mNbSel = 0;
   for (int aK=0 ; aK<int(mVRRC.size()) ; aK++)
   {
       if (mVSelec[aK])
       {
            mVSelec[mNbSel] = mVSelec[aK];
            mVIndex[mNbSel] = mVIndex[aK];
            mVRRC[mNbSel]   =   mVRRC[aK];
            mNbSel++;
       }
   }
   while (int(mVRRC.size()) > mNbSel)
   {
       mVSelec.pop_back();
       mVIndex.pop_back();
       mVRRC.pop_back();
   }
}

void cResulMultiImRechCorrel::SuprUnSelect(std::vector<cResulMultiImRechCorrel*> & aVR)
{
    int aNbSelGlob = 0;
    for (int aK=0  ; aK<int(aVR.size()) ; aK++)
    {
        aVR[aK]->SuprUnSelect();
        if (aVR[aK]->NbSel())
        {
            aVR[aNbSelGlob] = aVR[aK];
            aNbSelGlob++;
        }
        else
        {
            delete aVR[aK];
            aVR[aK] = 0;
        }
    }

    while (int(aVR.size()) > aNbSelGlob)
    {
        aVR.pop_back();
    }
}


    //==========================  cResulRechCorrel  ==================
    //==========================  cResulRechCorrel  ==================
    //==========================  cResulRechCorrel  ==================

cResulRechCorrel::cResulRechCorrel(const Pt2dr & aPt,double aCorrel)  :
     mPt     (aPt),
     mCorrel (aCorrel)
{
}

bool cResulRechCorrel::IsInit() const 
{
   return mCorrel > TT_DefCorrel;
}

cResulRechCorrel::cResulRechCorrel() :
   mPt     (0,0),
   mCorrel (TT_DefCorrel)
{
}

void cResulRechCorrel::Merge(const cResulRechCorrel & aRRC)
{
    if (aRRC.mCorrel > mCorrel)
    {
        // mCorrel = aRRC.mCorrel;
        // mPt     =  aRRC.mPt;
        *this = aRRC;
    }
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
aooter-MicMac-eLiSe-25/06/2007*/
