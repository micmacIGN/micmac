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


class cVecPrime
{

     public :
         typedef U_INT8 tInt;
         static cVecPrime  * FromCalc(int aNb,bool Exact,Pt2di & aInt);
         void Show(const Pt2di &Show) const;
         U_INT8  NbNombre() {return mV.back();}
         U_INT8  NbPrime() {return  mV.size();}
     private :
         bool  DivideByCur(tInt aVal) const;
         tInt  PushNext();


         cVecPrime();
         std::vector<tInt> mV;
         static      std::string  NameBuf();
};

std::string cVecPrime::NameBuf()
{
    return Basic_XML_User_File("BufPrime.dat");
}

cVecPrime::cVecPrime() 
{
}

bool  cVecPrime::DivideByCur(tInt aVal) const
{
   for (auto const & aV : mV)
   {
       if ((aVal%aV)==0)
          return true;
       if (aV*aV > aVal)
          return false;
   }
   ELISE_ASSERT(false,"DivideByCur");
   return true;
}

cVecPrime::tInt cVecPrime::PushNext()
{
    tInt  aV = mV.back() + 1;

    while (DivideByCur(aV))
      aV++;

   mV.push_back(aV);

   return aV;
       
}

void cVecPrime:: Show(const Pt2di &Show) const
{
    for (const auto & aV : mV)
    {
       if ((aV>= tInt(Show.x)) && (aV<= tInt(Show.y)))
          std::cout << "Prime : " << aV << "\n";
    }
   std::cout << "===================================\n";
}

cVecPrime *   cVecPrime::FromCalc(int aNb,bool Exact,Pt2di & aInterv)
{
    ElTimer aChrono;
    cVecPrime * aRes = new cVecPrime;
    if (ELISE_fp::exist_file(NameBuf()))
    {
       ELISE_fp aFp(NameBuf().c_str(),ELISE_fp::READ);
       int aNbFile = aFp.read_U_INT4();
       if (Exact)
       {
           aNbFile  = ElMin(aNb,aNbFile);
       }
          
       for (int aK=0 ; aK<aNbFile ; aK++)
       {
            aRes->mV.push_back(aFp.read_U_INT8());
       }
       aFp.close();
       aNb -= aNbFile;
    }
    else
    {
       aRes->mV.push_back(2);
    }

    ElTimer aT;

    for (int aK=1 ; aK< aNb ; aK++)
    {
        /*tInt aV =*/  aRes->PushNext();
    }
    //==================================
    if (!Exact)
    {
       ELISE_fp aFp(NameBuf().c_str(),ELISE_fp::WRITE);
       aFp.write_U_INT4(aRes->mV.size());
       for (int aK=0 ; aK<int(aRes->mV.size()) ; aK++)
       {
           aFp.write_U_INT8(aRes->mV[aK]);
       }
       aFp.close();
    }
    else
    {
    }

    if (EAMIsInit(&aInterv))
    {
         std::vector<tInt> aNewV;
    }
    

    std::cout << " Time=" << aChrono.uval() << "\n";

    double aNbP = aRes->NbPrime(); 
    double aNbN = aRes->NbNombre();
    std::cout << " NbN= " << aNbN  << " NbP=" << aNbP 
              << " Ratio=" <<  (aNbN/log(aNbN)) / aNbP
              << " Ratio=" <<  aNbN / (aNbP * log(aNbP))
               << "\n";

    return aRes;
}

int CPP_GenPrime(int argc,char** argv)
{
    int aNb;
    bool Exact=false;
    Pt2di aTestSeq;
    Pt2di aInterv;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNb,"Number of test on prime"),
        LArgMain()  << EAM(Exact,"Exact",true,"Exact number ?")
                    << EAM(aTestSeq,"Seq",true,"Sequence to test")
                    << EAM(aInterv,"Show",true,"Sequence to show")

    );
    cVecPrime *   aVP = cVecPrime::FromCalc(aNb,Exact,aInterv);

/*
    if (EAMIsInit(&aTestShow))
    {
       aVP->Show(aTestShow);
    }
*/



    delete aVP;
    return EXIT_SUCCESS;
}

void TestPrime()
{
    std::cout << "XXXX [" << Basic_XML_User_File("toto.tif") << "]\n";

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
