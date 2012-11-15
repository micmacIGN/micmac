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
#include "algo_geom/qdt_implem.h"


using namespace NS_ReducHoms;


/*************************************************/
/*                                               */
/*                XXXXXXX                        */
/*                                               */
/*************************************************/

void WarnTest()
{
    if (TEST)
    {
        for (int aK=0 ; aK< 10 ; aK++)
            std::cout << "========== !!!!! ATTENTION MODE TEST !!!!!!========\n";
    }
}

cAppliReduc::cAppliReduc(int argc,char ** argv) :
   mImportTxt    (false),
   mExportTxt    (false),
   mExtHomol     (""),
   mMinNbPtH     (20),
   mSeuilQual    (20),
   mRatioQualMoy (4.0),
   mKernConnec   (3),
   mKernSize     (6),
   mSetEq        (cNameSpaceEqF::eSysL2BlocSym)
   // mQT        (PtOfPhi,Box2dr(Pt2dr(-100,-100),Pt2dr(30000,30000)),10,500)
{
    
    CreateIndex();
   // Lecture bas niveau des parametres
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Directory (Dir+Pattern)"),
        LArgMain()  << EAM(mImportTxt,"ImpTxt",true,"Import in text format(def=false)")
                    << EAM(mExportTxt,"ExpTxt",true,"Export in text format(def=false)")
                    << EAM(mExtHomol,"ExtH",true,"Extension for homol, like SrRes, def=\"\")")
                    << EAM(mMinNbPtH,"NbMinHom",true,"Nb Min Pts For Homography Computation def=20")
                    << EAM(mSeuilQual,"SeuilQual",true,"Quality Theshold for homography (Def=20.0)")
                    << EAM(mRatioQualMoy,"RatioQualMoy",true,"Ratio to validate / average qual (def=4.0)")
    );

   SplitDirAndFile(mDir,mName,mFullName);

   // Creation noms et associations 
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mSetNameIm = mICNM->Get(mName);
   // mKeyHomol = "NKS-Set-Homol@"+mExtHomol+"@"+(mImportTxt ?"txt" : "dat");
   // mKeyH2I = "NKS-Assoc-CplIm2Hom@"+mExtHomol+"@"+(mImportTxt ?"txt" : "dat");
   mKeyHomol = KeyHIn("NKS-Set-Homol");
   mKeyH2I =  KeyHIn("NKS-Assoc-CplIm2Hom");
   mSetNameHom =  mICNM->Get(mKeyHomol);

   std::cout << "NbIm " << mSetNameIm->size() << " NbH " << mSetNameHom->size() << "\n";


   // Creation des images
   for (int aKIm=0 ; aKIm<int(mSetNameIm->size()) ; aKIm++)
   {
        const std::string & aName = (*mSetNameIm)[aKIm];
        cImagH * anIm=new cImagH(aName,*this,aKIm);
        mIms.push_back(anIm);
        mDicoIm[aName] = anIm;
   }
    

   // Creation des homologues
   for (int aKH=0 ; aKH<int(mSetNameHom->size()) ; aKH++)
   {
        const std::string & aName = (*mSetNameHom)[aKH];
        std::pair<std::string,std::string> aN1N2 = mICNM->Assoc2To1(mKeyH2I,aName,false);
        cImagH * aI1 = mDicoIm[aN1N2.first];
        cImagH * aI2 = mDicoIm[aN1N2.second];
        if (aI1 && aI2)
        {
            aI1->AddLink(aI2,aName);
        }
   }

}

std::string cAppliReduc::KeyHIn(const std::string & aKeyGen) const
{
    return aKeyGen+"@" + mExtHomol+"@"+(mImportTxt ?"txt" : "dat");
}



void cAppliReduc::ComputePts()
{
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
         mIms[aK]->ComputeLnkHom();
    }

    bool SpaceInit = true;

    // Init systeme equation
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
          cImagH * anI =  mIms[aK];
          anI->HF() = mSetEq.NewHomF(anI->Hi2t(),cNameSpaceEqF::eHomLibre);
    }
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
        cImagH * anI1 =  mIms[aK];
        const tSetLinks & aLL = anI1->Lnks();
        for (tSetLinks::const_iterator itL = aLL.begin(); itL != aLL.end(); itL++)
        {
            cImagH * anI2 = itL->second->Dest();
            itL->second->EqHF() = mSetEq.NewEqHomog(SpaceInit,*(anI1->HF()),*(anI2->HF()),0,false);
        }
    }

    int aNbBl = mSetEq.NbBloc();
    cAMD_Interf * mAMD = new cAMD_Interf (aNbBl);
    for (int aK=0 ; aK<aNbBl ; aK++)
    {
        mAMD->AddArc(aK,aK,true);
    }
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
        cImagH * anI1 =  mIms[aK];
        cHomogFormelle *  aHF1 = anI1->HF();

        const tSetLinks & aLL = anI1->Lnks();
        for (tSetLinks::const_iterator itL = aLL.begin(); itL != aLL.end(); itL++)
        {
            cImagH * anI2 = itL->second->Dest();
            cHomogFormelle *  aHF2 = anI2->HF();
            int aKbl1 = aHF1->IncInterv().NumBlocAlloc();
            int aKbl2 = aHF2->IncInterv().NumBlocAlloc();
            mAMD->AddArc(aKbl1,aKbl2,true);
        }
    }
   std::vector<int>  anOrder = mAMD->DoRank();
   const std::vector<cIncIntervale *>  & aVInt = mSetEq.BlocsIncAlloc();
   for (int aK=0 ; aK<int(aVInt.size()) ; aK++)
   {
       aVInt[aK]->SetOrder(anOrder[aK]);
   }

   mSetEq.SetClosed();

    // Init Noyau
    TestMerge();

    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
         ClearIndex();
         mIms[aK]->ComputePts();
    }
}

cSetEqFormelles &  cAppliReduc::SetEq()
{
   return mSetEq;
}

const std::string & cAppliReduc::Dir() const
{
   return mDir;
}



void cAppliReduc::DoAll()
{
    ComputePts();
    cPtHom::ShowAll();
}

int  cAppliReduc::MinNbPtH() const
{
   return mMinNbPtH;
}


double cAppliReduc::SeuilQual () const
{
   return mSeuilQual;
}

double cAppliReduc::RatioQualMoy () const
{
   return mRatioQualMoy;
}

int    cAppliReduc::KernConnec() const
{
    return mKernConnec;
}
int    cAppliReduc::KernSize() const
{
    return mKernSize;
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
