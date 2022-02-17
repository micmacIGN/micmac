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
         typedef U_INT8 tInt;  // Type of storage for number

         /*  Allocate the vector of prim :
               * Nb : number of prime
         */

         static cVecPrime  * FromCalc(int & aNb,Pt2di & aInt);
         void MakeOneFamillyTest(const std::vector<int>&,bool Show) const;
         // void MakeFamillyTest(const std::vector<int>&,bool Show) const;


         void Show(const Pt2di &Show) const;
         tInt  NbNombre()      const {return mVPrimes.back();}
         tInt  NbPrime()       const {return  mVPrimes.size();}
         tInt  Prime(tInt aK)  const {return  mVPrimes.at(aK);}
     private :
         bool OkOneFamillyTest(int aK,const std::vector<int> &,bool Show) const;

         bool  DivideByCur(tInt aVal) const;
         tInt  PushNext();


         cVecPrime();
         std::vector<tInt> mVPrimes;
         static      std::string  NameBuf();
};

/*  Familly conjecture , one familly */
bool cVecPrime::OkOneFamillyTest(int aKPrim,const std::vector<int> & aVDif,bool Show) const
{
    if ( (aKPrim+aVDif.size()>=mVPrimes.size()))
      return false;

    tInt aP0 = mVPrimes[aKPrim];
    for (unsigned int aKDif=0 ; aKDif<aVDif.size(); aKDif++)
    {
        if (mVPrimes[aKPrim+aKDif+1] != aP0 + aVDif[aKDif])
           return false;
    }
    if (Show)
    {
        std::cout << "P : " << aP0;
        for (unsigned int aKDif=0 ; aKDif<aVDif.size(); aKDif++)
            std::cout << " " << mVPrimes[aKPrim+aKDif+1]  ;
        std::cout << "\n";
    }

    return true;
}

void cVecPrime::MakeOneFamillyTest(const std::vector<int> & aVDif,bool Show) const
{
   int aNbFam=0;
   for (unsigned int aKPrim=0 ; aKPrim<mVPrimes.size()  ; aKPrim++)
   {
       if (OkOneFamillyTest(aKPrim,aVDif,Show))
          aNbFam++;
   }

   double aNbN = NbNombre();
   double aTh = aNbN / pow(log(aNbN),1+aVDif.size());

   std::cout << "NbF=" << aNbFam << " Ratio "  << aNbFam/aTh << "\n";

}





std::string cVecPrime::NameBuf()
{
    return Basic_XML_User_File("BufPrime.dat");
}

cVecPrime::cVecPrime() 
{
}

bool  cVecPrime::DivideByCur(tInt aVal) const
{
   for (auto const & aV : mVPrimes)
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
    tInt  aV = mVPrimes.back() + 1;

    while (DivideByCur(aV))
      aV++;

   mVPrimes.push_back(aV);

   return aV;
       
}

void cVecPrime:: Show(const Pt2di &Show) const
{
    for (const auto & aV : mVPrimes)
    {
       if ((aV>= tInt(Show.x)) && (aV<= tInt(Show.y)))
          std::cout << "Prime : " << aV << "\n";
    }
   std::cout << "===================================\n";
}

cVecPrime *   cVecPrime::FromCalc(int & aNb,Pt2di & aIntervShow)
{
    ElTimer aChrono;
    cVecPrime * aRes = new cVecPrime;
    bool gotChange = false;  // Will we need to save changes
    U_INT4 aSzFile=0;
    // If the file already exist
    if (ELISE_fp::exist_file(NameBuf()))
    {
       ELISE_fp aFp(NameBuf().c_str(),ELISE_fp::READ);
       aSzFile = aFp.read_U_INT4(); // Number of int already created
       int aNbFile = aSzFile;
       if (aNb !=-1)
       {
           // If exact, truncate if too many
           aNbFile  = ElMin(aNb,aNbFile);
       }
       else
       {
           aNb =  aNbFile;
       }
          
       for (int aK=0 ; aK<aNbFile ; aK++)
       {
            aRes->mVPrimes.push_back(aFp.read_U_INT8());
       }
       aFp.close();
       aNb -= aNbFile;
    }
    else
    {
       gotChange = true;
       aNb --;
       aRes->mVPrimes.push_back(2);
    }

    ElTimer aT;

    for (int aK=0 ; aK< aNb ; aK++)
    {
         gotChange = true;
         aRes->PushNext();
    }
    //==================================
    if (gotChange)
    {
       ELISE_fp aFp(NameBuf().c_str(),ELISE_fp::WRITE);
       aFp.write_U_INT4(aRes->mVPrimes.size());
       for (int aK=0 ; aK<int(aRes->mVPrimes.size()) ; aK++)
       {
           aFp.write_U_INT8(aRes->mVPrimes[aK]);
       }
       aFp.close();
    }
    else
    {
    }

    if (EAMIsInit(&aIntervShow))
    {
         std::cout << "SHOW FIRST PRIME \n ";
         for (tInt aK=aIntervShow.x ; aK<tInt(aIntervShow.y) ; aK++)
         {
             if (aK<aRes->NbPrime())
                std::cout << aRes->Prime(aK) << " ";
         }
         std::cout << "\n=======================\n ";
    }
    

    std::cout << " Time=" << aChrono.uval() << "\n";

    double aNbP = aRes->NbPrime(); 
    double aNbN = aRes->NbNombre();
    std::cout  << " NbN= " << aNbN  << " NbP=" << aNbP 
               << " Ratio=" <<  (aNbN/log(aNbN)) / aNbP
               << " Ratio=" <<  aNbN / (aNbP * log(aNbP))
               << " SzFile=" <<  aSzFile
               << " Change=" <<  gotChange
               << "\n";

    return aRes;
}

int CPP_GenPrime(int argc,char** argv)
{
    int aNb;
    Pt2di aTestSeq;
    Pt2di aIntervShow;
    std::vector<int>  aFamTest;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNb,"Number of test on prime"),
        LArgMain()  << EAM(aTestSeq,"Seq",true,"Sequence to test")
                    << EAM(aIntervShow,"Show",true,"Sequence to show")
                    << EAM(aFamTest,"FamSeq",true,"Familly (twin, cousin ...) sequnce to test [Ofs1,Ofs2,...] ")

    );
    cVecPrime *   aVP = cVecPrime::FromCalc(aNb,aIntervShow);

    if (EAMIsInit(&aFamTest))
    {
       bool ShowFam = false;
       aVP->MakeOneFamillyTest(aFamTest,ShowFam);
    }

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
