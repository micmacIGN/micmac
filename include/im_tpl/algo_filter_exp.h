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



#ifndef _ELISE_IM_ALGO_FILTER_EXP
#define _ELISE_IM_ALGO_FILTER_EXP

template <class TypeI,class TypeF>  void  FilterLinExpVar
                     (
                           TypeI * input,
                           const TypeF * Fact,
                           double * aBuf,
                           int      aNb
                     )
{
   aBuf[0] = 0;
   for (int anX = 1; anX<aNb ; anX++)
   {
       aBuf[anX] =  Fact[anX-1] *(aBuf[anX-1] + input[anX-1]);
   }

   for (int anX = aNb-2; anX>=0 ; anX--)
   {
           input[anX]   += Fact[anX+1] * input[anX+1];
           input[anX+1] += (TypeI)( aBuf[anX+1] );
   }
}

template <class TypeI,class TypeF>  void  FilterMoyenneExpVar
                     (
                           TypeI * input,
                           const TypeF * Fact,
                           int      aNb,
                           int      aNbIter
                     )
{
/*
static int aCpt=0; aCpt++;
double aV = input[aNb/2];
bool Bug = (aCpt==641);

if (Bug)
{
     for (int anX = 0; anX<aNb ; anX++)
     {  
            if (isnan(input[anX]) || isnan(Fact[anX]))
            {
                std::cout << "AAAAAAAAAA " << anX << "\n"; getchar();
            }
     }  
}
*/


     Im1D_REAL8 aIBuf(aNb);
     double * aBuf = aIBuf.data();

     Im1D_REAL4 aIPond(aNb,1.0);
     float *   aDPond =  aIPond.data();

     FilterLinExpVar(aDPond,Fact,aBuf,aNb); 
     for (int anX = 0; anX<aNb ; anX++)
     {
        aDPond[anX] = (float)( ElMax(1e-4,double(aDPond[anX])) );
     }
     for (int aKIt = 0 ; aKIt<aNbIter ; aKIt ++)
     {
          FilterLinExpVar(input,Fact,aBuf,aNb);
          for (int anX = 0; anX<aNb ; anX++)
          {
               input[anX] /= aDPond[anX];
          }
     }
/*
std::cout <<  "HHHHH " << aV << " " <<  input[aNb/2] << "\n";
if( isnan(input[aNb/2]))
{
    std::cout << "CPT nana " << aCpt << "\n"; getchar();
}
*/
}



template <class T1> void  FilterExp
                          (
                               T1 & anIm,
                               const Box2di &  aBox,
                               double aFx,
                               double aFy
                          )
{
    Pt2di aSz = anIm.sz();
    int aNbBuf = 2 + aSz.x + aSz.y;
    Im1D_REAL8 mBufL(aNbBuf,0.0);

    typename T1::tElem **  aD = anIm.data();
    double  * mBL = mBufL.data();

    int anX0 = aBox._p0.x;
    int anX1 = aBox._p1.x;
    int anY0 = aBox._p0.y;
    int anY1 = aBox._p1.y;


    for (int anY=anY0 ; anY<anY1 ; anY++)
    {
        typename T1::tElem * aLine = aD[anY];
        mBL[anX0] = 0;
        for (int anX = anX0+1; anX<anX1 ; anX++)
        {
           mBL[anX] =  aFx *(mBL[anX-1] + aLine[anX-1]);
        }

        for (int anX = anX1-2; anX>=anX0 ; anX--)
        {
           aLine[anX]   += (typename T1::tElem)( aFx*aLine[anX+1] );
           aLine[anX+1] += (typename T1::tElem)( mBL[anX+1] );
        }
    }
    for (int anX=anX0 ; anX<anX1 ; anX++)
    {
        mBL[anY0] = 0;
        for (int anY = anY0+1; anY<anY1 ; anY++)
        {
           mBL[anY] =  aFy *(mBL[anY-1] + aD[anY-1][anX]);
        }
        for (int anY = anY1-2; anY>=anY0 ; anY--)
        {
           aD[anY][anX]   += (typename T1::tElem)( aFy * aD[anY+1][anX] );
           aD[anY+1][anX] += (typename T1::tElem)( mBL[anY+1] );
        }
    }
}

template <class T1> void  FilterExp (T1 & anIm, double aFx, double aFy)
{
     FilterExp(anIm,Box2di(Pt2di(0,0),anIm.sz()),aFx,aFy);
}

template <class T1> void  FilterExp (T1 & anIm, double aFxy)
{
     FilterExp(anIm,aFxy,aFxy);
}



template <class T1> void  FilterGauss(T1 & anIm, double aSzF,int aNbIter = 4)
{
  double aF = FromSzW2FactExp(aSzF,aNbIter);

  Pt2di aSz = anIm.sz();
  Im2D_REAL4 aIP1(aSz.x,aSz.y,1.0);
  FilterExp(aIP1,aF);

  for (int aKIt=0 ; aKIt<aNbIter ; aKIt++)
  {
      FilterExp(anIm,aF);
      ELISE_COPY(anIm.all_pts(),anIm.in()/aIP1.in(),anIm.out());
  }
}

template <class T1,class T2> void  MasqkedFilterGauss(T1 & anIm, T2& aMasq,double aSzF,int aNbIter = 4)
{
  double aF = FromSzW2FactExp(aSzF,aNbIter);

  Pt2di aSz = anIm.sz();
  Im2D_REAL4 aIP1(aSz.x,aSz.y);
  ELISE_COPY(aIP1.all_pts(),aMasq.in(),aIP1.out());
  FilterExp(aIP1,aF);
  ELISE_COPY(aIP1.all_pts(),Max(1e-5,aIP1.in()),aIP1.out());

  for (int aKIt=0 ; aKIt<aNbIter ; aKIt++)
  {
      ELISE_COPY(anIm.all_pts(),anIm.in()*aMasq.in(),anIm.out());
      FilterExp(anIm,aF);
      ELISE_COPY(anIm.all_pts(),anIm.in()/aIP1.in(),anIm.out());
  }
}





#endif  //  _ELISE_IM_ALGO_FILTER_EXP











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
