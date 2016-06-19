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

#include "Xeres.h"


/***************************************************************/
/*                                                             */
/*             cXeres_NumSom                                   */
/*                                                             */
/***************************************************************/

cXeres_NumSom::cXeres_NumSom(int aNumParal,int aNumMer,double aTeta,const std::string & aName,bool aPorte) :
   mName        (aName),
   mNumParal    (aNumParal),
   mNumMer      (aNumMer),
   mTeta        (mod_real(aTeta,2*PI)),
   mDirT        (Pt2dr::FromPolar(1,mTeta)),
   mPorte       (aPorte)
{
   // std::cout << mName  << "Par=" << mNumParal << " Mer=" << mNumMer << " T=" << mTeta << " P=" << mPorte <<  "\n";
}

const std::string & cXeres_NumSom::Name() const {return mName;}
const int & cXeres_NumSom::NumParal() const {return mNumParal;}
const int & cXeres_NumSom::NumMer() const   {return mNumMer;}
const double & cXeres_NumSom::Teta() const  {return mTeta;}
const Pt2dr & cXeres_NumSom::DirT() const   {return mDirT;}
const bool & cXeres_NumSom::Porte() const   {return mPorte;}


cXeres_NumSom cXeres_NumSom::CreateFromStdName_V0(const std::string & aName)
{
    char aC = aName[0];
    int aNum ;
    bool aPorte = false;
    FromString(aNum,aName.c_str()+1);


    // A1 -A7  B1-B7 ....
    if ((aC>='A') && (aC<'X'))
    {
        int aNumPar =  8-aNum;
        
        double  aNumMer  = aC - 'A';
        if (aC <= 'L')
        {
        }
        else if (aC<='O')
        {
            aPorte = true;
            aNumMer += (aC- 'L') / 3.0;
        }
        else if (aC<='W')
        {
               aNumMer += 1;  // Y'a un trou dans la numerotation
        }
        else
        {
              ELISE_ASSERT(false,"Un handled case in cXeres_NumSom::CreateFromStdName_V0");
        }
        double aTeta = (aNumMer+0.5) * ((2*PI) /24.0);

/*  
       // Verif Teta
        if (aNum==1)
        {
            std::cout << "For " << aC << " mer= " << aNumMer << "\n";
        }
*/
        return cXeres_NumSom(aNumPar,round_ni(aNumMer),aTeta, aName,aPorte);
    }


    if (aC== 'X')
    {
         // X1 - XX18
         if (aNum<= 18)
         {
             if (aNum> 15)
             {
                 aPorte = true;
             }
             int aNumPar = 8;
             // X7 est +ou- equiv a A1
             double aTeta = (aNum+0.5-7) * ((2*PI) / 18.0);
             return cXeres_NumSom(aNumPar,aNum,aTeta, aName,aPorte);
         }
         else if (aNum<=30)
         {
             if (aNum> 28)
             {
                 aPorte = true;
             }
             int aNumPar = 9;
             
             // X19 est +ou- equiv a A1
             double aTeta = (aNum+0.5-19) * ((2*PI) / 12.0);
             return cXeres_NumSom(aNumPar,aNum,aTeta, aName,aPorte);
             
         }
         else
         {
              ELISE_ASSERT(false,"Un handled case in cXeres_NumSom::CreateFromStdName_V0");
         }
    }

   
    std::cout << "FOR NAME= " << aName << "\n";
    ELISE_ASSERT(false,"cXeres_NumSom::CreateFromStdName_V0 : cas non geree ");
    return cXeres_NumSom(0,0,0, aName,false);
}

   
/***************************************************************/
/*                                                             */
/*             cAppliXeres                                     */
/*                                                             */
/***************************************************************/

class cCmp_XCPtr_TmpDist
{
    public :
        bool operator ()(cXeres_Cam * aC1,cXeres_Cam * aC2)
        {
            return aC1->TmpDist() < aC2->TmpDist() ;
        }
};


std::vector<cXeres_Cam *> cAppliXeres::GetNearestNeigh(cXeres_Cam *aCam0,int aDL,int aNb)
{
     std::vector<cXeres_Cam *>  aVCdt;
     const cXeres_NumSom & aNS0 = aCam0->NS();
     int aParal = aNS0.NumParal() + aDL;
     double aTeta0 =  aNS0.Teta();

     for (int aKC=0 ; aKC<int(mVCam.size()) ; aKC++)
     {
          cXeres_Cam * aCamVois = mVCam[aKC];
          const cXeres_NumSom & aNSV = aCamVois->NS();
          
          if (aNSV.NumParal() == aParal)
          {
              if (aCamVois==aCam0)
              {
              }
              else
              {
                 double aDTeta = Centered_mod_real(aTeta0-aNSV.Teta(),2*PI) ;
                 
                 aCamVois->TmpDist() = ElAbs(aDTeta);
                 aVCdt.push_back(aCamVois);
              }
          }
     }

     cCmp_XCPtr_TmpDist aCmp;
     std::sort(aVCdt.begin(),aVCdt.end(),aCmp);

/*
std::cout << "HHHH " << aCam0->NS().Name() << " P=" <<  aParal << " DL=" << aDL <<  " Got=" << aVCdt.size() <<  " Req=" << aNb << "\n";
for (int aK=0 ; aK<5 ; aK++)
{
   std::cout << aVCdt[aK]->NS().Name() << " " << aVCdt[aK]->TmpDist() << "|";
}
std::cout << "\n";
*/

     return  std::vector<cXeres_Cam *>
             (
                  aVCdt.begin(),
                  aVCdt.begin() + ElMin(aNb,int(aVCdt.size()))
             );
}

std::vector<cXeres_Cam *> cAppliXeres::GetNearestExistingNeigh(cXeres_Cam *aCam0,int aDL,int aNb)
{
    ELISE_ASSERT(aCam0->HasIm(),"cAppliXeres::GetNearestExistingNeigh no Im");
    std::vector<cXeres_Cam *> aVCdt = GetNearestNeigh(aCam0,aDL,aNb);

    std::vector<cXeres_Cam *> aVRes;
    for (int aKC=0 ; aKC<int(aVCdt.size()) ; aKC++)
    {
         if (aVCdt[aKC]->HasIm())
           aVRes.push_back(aVCdt[aKC]);
    }
    return aVRes;
}


void  cAppliXeres::AddNearestExistingNeigh(std::vector<cXeres_Cam *> & aRes,cXeres_Cam *aCam0,int aDL,int aNb)
{
    std::vector<cXeres_Cam *> aVAdd = GetNearestExistingNeigh(aCam0,aDL,aNb);
    std::copy ( aVAdd.begin(), aVAdd.end(), back_inserter(aRes));
}

std::vector<cXeres_Cam *> cAppliXeres::NeighVois(cXeres_Cam * aCam,int aDeltaV)
{
   std::vector<cXeres_Cam *> aRes ;
   for (int aK=-aDeltaV ; aK<=aDeltaV ; aK++)
   {
      int aNb = 1 + 2 * ElAbs(ElAbs(aK)-aDeltaV);
      if (aK==0) aNb--;

      AddNearestExistingNeigh(aRes,aCam,aK,aNb);
   }
   return aRes;
}

std::vector<cXeres_Cam *> cAppliXeres::NeighPtsHom(cXeres_Cam * aCam)
{
   return NeighVois(aCam,2);
}

std::vector<cXeres_Cam *> cAppliXeres::NeighMatch(cXeres_Cam * aCam)
{
   return NeighVois(aCam,1);
}


void  cAppliXeres::TestOneNeigh(const std::string & aName,int aDeltaV)
{
    cXeres_Cam * aCam = mMapCam[aName];
    if (aCam==0)
    {
        std::cout << "Id is not valid : " << aName << "\n\n";
    }
    else if (! aCam->HasIm())
    {
        std::cout << "Cam has no image : " << aName << "\n\n";
    }
    else
    {
         std::vector<cXeres_Cam *> aVC = NeighVois(aCam,aDeltaV);
         for (int aK=0 ; aK<int(aVC.size()) ; aK++)
             std::cout << aVC[aK]->NS().Name() << " " ;
          std::cout << "\n\n";
    }
}

void  cAppliXeres::TestInteractNeigh()
{
     while (1)
     {
         std::string anId;
         int aNbV;
         std::cout << "ENTER Id/NbV \n";
         std::cin >> anId >> aNbV ;
         TestOneNeigh(anId,aNbV);
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
