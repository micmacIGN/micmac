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


#include "NewRechPH.h"

std::vector<int> VectScal(const cOnePCarac * aP1,const cOnePCarac * aP2)
{
    Im2D_INT1 aI1 =  aP1->InvR().ImRad();
    Im2D_INT1 aI2 =  aP2->InvR().ImRad();
    int aTy = aI1.ty();
    int aTx = aI1.tx();

    std::vector<int> aRes;
    int aScalGlob = 0;
    for (int aY=0 ; aY<aTy ; aY++)
    {
       int aScal = 0;
       INT1 *aD1 = aI1.data()[aY];
       INT1 *aD2 = aI2.data()[aY];
       for (int aX=0 ; aX<aTx ; aX++)
       {
          aScal += aD1[aX] * aD2[aX];
       }
       aRes.push_back(aScal);
       aScalGlob += aScal;
    }
    aRes.push_back(aScalGlob);
    return aRes;
}

Im2D_INT1 Get_IR0(const cRotInvarAutoCor & aRIAC){return aRIAC.IR0();}
Im2D_INT1 Get_IGT(const cRotInvarAutoCor & aRIAC){return aRIAC.IGT();}
Im2D_INT1 Get_IGR(const cRotInvarAutoCor & aRIAC){return aRIAC.IGR();}
typedef Im2D_INT1 (*tGet_I1FromRIAC)(const cRotInvarAutoCor & aRIAC);


double ScoreI1(tGet_I1FromRIAC aFonc,const cRotInvarAutoCor & aRIAC1,const cRotInvarAutoCor & aRIAC2)
{
    Im2D_INT1 aI1 = aFonc(aRIAC1);
    Im2D_INT1 aI2 = aFonc(aRIAC2);
    int aSDif,aSom;
    ELISE_COPY
    (
        aI1.all_pts(),
        Virgule(1,Abs(aI1.in()-aI2.in())),
        Virgule(sigma(aSom),sigma(aSDif))
    );

    return (aSDif / double(aSom)) / DynU1;
}

std::vector<double> GetVSI1(const cRotInvarAutoCor & aRIAC1,const cRotInvarAutoCor & aRIAC2)
{
    std::vector<tGet_I1FromRIAC> aVF={Get_IR0,Get_IGT,Get_IGR};
    std::vector<double> aRes;
    double aSom=0;
    for (const auto & aFonc : aVF)
    {
        double aSc = ScoreI1(aFonc,aRIAC1,aRIAC2);
        aRes.push_back(aSc);
        aSom += aSc;
    }
    aRes.push_back(aSom/aRes.size());
    return aRes;
}


void TestOneInvR
     (
         const std::vector<cOnePCarac> & aVH,
         const cOnePCarac * aHom1,
         const cOnePCarac * aHom2
     )
{
    std::vector<double>  aVD0 = GetVSI1(aHom1->RIAC(),aHom2->RIAC());
    int aNbLab = aVD0.size();
    std::vector<int>     aVNb(aNbLab,0);
    std::vector<int>     aVOk(aNbLab,0);
    std::vector<double>  aVDMoy(aNbLab,0.0);


    for (int aK=0 ; aK<int(aVH.size()) ; aK++)
    {
         const cOnePCarac * aHomTest = & aVH[aK];
         if (aHomTest->Kind() == aHom2->Kind())
         {
             
             std::vector<double>  aVDist  = GetVSI1(aHomTest->RIAC(),aHom2->RIAC());
             for (int aK=0 ; aK<aNbLab ; aK++)
             {
                  aVNb[aK] ++;
                  aVDMoy[aK] += aVDist[aK];
                  if (aVD0[aK] <= aVDist[aK])
                     aVOk[aK] ++;
             }
         }
    }

    std::cout <<  "========= AUTO CORREL ==========\n";
    for (int aK=0 ; aK<aNbLab ; aK++)
    {
         std::cout << " Prop " << (aVOk[aK]/double(aVNb[aK])) 
                   << " DRef " << aVD0[aK] 
                   << " DMoy " << aVDMoy[aK] / aVNb[aK]
                   << " A " << aVOk[aK] << " " << aVNb[aK]
                   << "\n";
    }
}


double ScoreTestMatchInvRad(const std::vector<cOnePCarac> & aVH,const cOnePCarac * aHom1,const cOnePCarac * aHom2,bool Show)
{
     ELISE_ASSERT(aHom1->Kind() == aHom2->Kind(),"Dif Kind in TestMatchInvRad");
     
     std::vector<int> aVS0 =  VectScal(aHom1,aHom2);
     std::vector<int> aVNbOk(aVS0.size(),0);
     int aNbOkLab = 0;
     for (int aK=0 ; aK<int(aVH.size()) ; aK++)
     {
         const cOnePCarac * aHomTest = & aVH[aK];
         if (aHomTest->Kind() == aHom2->Kind())
         {
             aNbOkLab ++;
             std::vector<int> aVScal =  VectScal(aHomTest,aHom2);
             for (int aK=0 ; aK<int(aVScal.size()); aK++)
                 aVNbOk[aK]  += (aVScal[aK] < aVS0[aK]);
         }
     }
     if (Show)
     {
        for (int aK=0 ; aK<int(aVNbOk.size()) ; aK++)
           std::cout << "fFor " <<  eToString(eTypeInvRad(aK))  << " " << aVNbOk[aK] / double(aNbOkLab) << "\n";
        TestOneInvR(aVH,aHom1,aHom2);
     }

     return aVNbOk.back() / double(aNbOkLab);
}

void TestMatchInvRad(const std::vector<cOnePCarac> & aVH,const cOnePCarac * aHom1,const cOnePCarac * aHom2)
{
     ScoreTestMatchInvRad(aVH,aHom1,aHom2,true);
}


double ScoreTestMatchInvRad(const std::vector<cOnePCarac> & aVH,const cOnePCarac * aHom1,const cOnePCarac * aHom2)
{
   return ScoreTestMatchInvRad(aVH,aHom1,aHom2,false);
}

#if (0)

class cInterfNapp3D
{
    public :
        virtual void SetVal(const Pt2di &,const double & aZ,double aVal) = 0;
    private :
};
#endif


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
