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
#include "Digeo.h"

namespace NS_ParamDigeo
{


/****************************************/
/*                                      */
/*           cTplImInMem                */
/*                                      */
/****************************************/


template <class Type> typename cTplImInMem<Type>::tBase cTplImInMem<Type>::DOG(Type *** aC,const Pt3di& aP)
{
     return aC[aP.z][aP.y][aP.x]-aC[aP.z+1][aP.y][aP.x];
}

template <class Type> bool cTplImInMem<Type>::SupDOG(Type *** aC,const Pt3di& aP1,const Pt3di& aP2)
{
    tBase aV1 = DOG(aC,aP1);
    tBase aV2 = DOG(aC,aP2);

    if (aV1 > aV2) return true;
    if (aV1 < aV2) return false;

    if (aP1.z > aP2.z) return true;
    if (aP1.z < aP2.z) return false;

    if (aP1.y > aP2.y) return true;
    if (aP1.y < aP2.y) return false;

    if (aP1.x > aP2.x) return true;
    if (aP1.x < aP2.x) return false;

    ELISE_ASSERT(false,"::SupDOG");
    return true;
}

template <class Type> void cTplImInMem<Type>::ExtramDOG(Type *** aC,const Pt2di & aP,bool & isMax,bool & isMin)
{
    isMax = true;
    isMin = true;

    for (int aZ=-1 ; aZ<=1 ; aZ++)
    {
       for (int aY=-1 ; aY<=1 ; aY++)
       {
          for (int aX=-1 ; aX<=1 ; aX++)
          {
              if (aZ||aY||aX)
              {
                    if (SupDOG(aC,Pt3di(aP.x,aP.y,0),Pt3di(aP.x+aX,aP.y+aY,aZ)))
                       isMin = false;
                    else
                       isMax = false;
                    if ((!isMin) && (!isMax))
                       return;
              }
          }
       }
    }
}


template <class Type> 
eTypeExtreSift cTplImInMem<Type>::CalculateDiff
     (
          Type *** aC,
          int      anX,
          int      anY,
          int      aNiv
     )
{
    Type ** aI = aC[0] +anY;
    Type* aIm1 = aI[-1]+anX;
    Type* aI0  = aI[ 0]+anX;
    Type* aIp1 = aI[ 1]+anX;


    Type ** aJ = aC[1] +anY;
    Type* aJm1 = aJ[-1]+anX;
    Type* aJ0  = aJ[ 0]+anX;
    Type* aJp1 = aJ[ 1]+anX;


    theMDog[-1][-1] = aIm1[-1]-aJm1[-1];
    theMDog[-1][ 0] = aIm1[ 0]-aJm1[ 0];
    theMDog[-1][ 1] = aIm1[ 1]-aJm1[ 1];

    theMDog[0][-1] = aI0[-1]-aJ0[-1];
    theMDog[0][ 0] = aI0[ 0]-aJ0[ 0];
    theMDog[0][ 1] = aI0[ 1]-aJ0[ 1];

    theMDog[ 1][-1] = aIp1[-1]-aJp1[-1];
    theMDog[ 1][ 0] = aIp1[ 0]-aJp1[ 0];
    theMDog[ 1][ 1] = aIp1[ 1]-aJp1[ 1];


    mGX =  (theMDog[0][1]-theMDog[0][-1])/2.0;
    mGY =  (theMDog[1][0]-theMDog[-1][0])/2.0;

    mDxx = theMDog[0][1]+theMDog[0][-1] - 2*theMDog[0][0];
    mDyy = theMDog[1][0]+theMDog[-1][0] - 2*theMDog[0][0];

    mDxy = (theMDog[1][1]+theMDog[-1][-1]-theMDog[1][-1]-theMDog[-1][1])/4.0;

    double aDelta = mDxx * mDyy - ElSquare(mDxy);

    if (aDelta<=0) 
        return eTES_instable;

    mTrX = - ( mDyy*mGX-mDxy*mGY) / aDelta;
    mTrY = - (-mDxy*mGX+mDyy*mGY) / aDelta;

    int aDx = round_ni(mTrX);
    int aDy = round_ni(mTrY);


    if ((aDx!=0) || (aDy!=0))
    {
         if (aNiv>=3) 
         {
            return eTES_instable;
         }

         anX += aDx;
         anY += aDy;

         if (     (anX>=mBrd)
              &&  (anX<mSz.x-mBrd)
              &&  (anY>=mBrd)
              &&  (anY<mSz.y-mBrd)
            )
         {
             return CalculateDiff(aC,anX,anY,aNiv+1);
         }
         else
            return eTES_instable;
    }


    double aTrace = mDxx + mDyy;
    double aRatio = ElSquare(aTrace) / aDelta;

    if (ElSquare(mGX)+ElSquare(mGY)<mSeuilGrad)
       return eTES_GradFaible;

    if (aRatio > mSeuilTr2Det)
       return eTES_TropAllonge;

/*
std::cout << sqrt((ElSquare(mGX)+ElSquare(mGY))/mSeuilGrad) << " " 
          << " " << mTrX
          << " " << mTrY
          << " " << aNiv
          << " Ratio : " << aRatio
          << "\n";
*/

   return eTES_Ok;
}




template <class Type> 
void cTplImInMem<Type>::ExtractExtremaDOG
     (
          const cSiftCarac & aSC,
          cTplImInMem<Type> & aPrec,
          cTplImInMem<Type> & aNext1,
          cTplImInMem<Type> & aNext2
     )
{

   double aRalm = aSC.RatioAllongMin().Val();
   mSeuilTr2Det = (aRalm+1)*(1+1/aRalm);
   mSeuilGrad = mImGlob.G2Moy() * ElSquare( aSC.RatioGrad().Val() /mResolOctaveBase);

   if (theMDog==0)
   {
      theMDog = NEW_MATRICE(Pt2di(-1,-1),Pt2di(2,2),tBase);
   }


   cVisuCaracDigeo * aVCD = mImGlob.CurVisu();
   Type *** aC = mTOct.Cube() + mKInOct;
   bool aVerifExtrema =     mAppli.SectionTest().IsInit()
                        &&  mAppli.SectionTest().Val().VerifExtrema().Val();

   if (aVerifExtrema)
   {
       std::vector<float>  aV;
       for (int anX=5 ; anX < (mSz.x-5) ; anX+=5)
       {
          for (int anY=5 ; anY < (mSz.y-5) ; anY+=5)
          {
              double aD1 = aC[-1][anY][anX] -aC[0][anY][anX];
              double aD2 = aC[0][anY][anX] -aC[1][anY][anX];

              aV.push_back(ElAbs(aD1)/(1e-5+ElAbs(aD2)));
          }
       }

       std::sort(aV.begin(),aV.end());
       if (mKInOct==1) std::cout << "**********************************\n";
       std::cout << "MDIAN  [" << mKInOct << "]=" 
                 << aV[aV.size()/10] << " ; "
                 << aV[aV.size()/2] << " ; "
                 << aV[(9*aV.size())/10] << "\n";

       while (0)
       {
           int x,y;
           cin >> x >> y;

           std::cout << aC[-1][y][x] << " " << aC[0][y][x] << " " << aC[1][y][x] << "\n";
       }
   }

   bool aInteract = aVerifExtrema || (aVCD!=0);

   ELISE_ASSERT
   (
       (mSz==aPrec.mSz)&&(mSz==aNext1.mSz)&&(mSz==aNext2.mSz),
       "Size im diff in ::ExtractExtremaDOG"
   );

   Im1D<tBase,tBase> aImDif(mSz.x);
   mNbExtre = 0;
   mNbExtreOK = 0;


  bool doMax = aSC.DoMax().Val();
  bool doMin = aSC.DoMin().Val();


  mBrd = 1;
  for (int anY=mBrd ; anY<(mSz.y-mBrd) ; anY++)
  {

        tBase * aLDif = aImDif.data();


       Type * aLm1 = aC[0][anY-1];
       Type * aL = aC[0][anY];
       Type * aLp1 = aC[0][anY+1];


        Type * aN1m1 =  aC[1][anY-1];
        Type * aN1   =  aC[1][anY];
        Type * aN1p1 =  aC[1][anY+1];


        for (int anX = 0; anX<mSz.x ; anX++)
        {
            aLDif[anX] = aL[anX] -aN1[anX];
        }

        aLDif+= mBrd;
        
        int aX1 = mSz.x-mBrd;

        for (int anX = mBrd; anX<aX1 ; anX++)
        {
           mDogPC = *aLDif;
           bool isMax=false;
           bool isMin=false;

           if (mDogPC>=aLDif[-1])
           {
              if (      doMax
                    &&  (mDogPC> aLDif[1])
                    &&  (mDogPC>=  (aLm1[anX]-aN1m1[anX]))
                    &&  (mDogPC>   (aLp1[anX]-aN1p1[anX]))
                    &&  (mDogPC>=  (aC[-1][anY][anX] -aC[0][anY][anX]))
                    &&  (mDogPC>   (aC[1][anY][anX] -aC[2][anY][anX]))


                    &&  (mDogPC>=  aLm1[anX+1]-aN1m1[anX+1])
                    &&  (mDogPC>=  aLm1[anX-1]-aN1m1[anX-1])
                    &&  (mDogPC>   aLp1[anX+1]-aN1p1[anX+1])
                    &&  (mDogPC>   aLp1[anX-1]-aN1p1[anX-1])

                    &&  (mDogPC>= (aC[-1][anY-1][anX-1] -  aC[0][anY-1][anX-1]))
                    &&  (mDogPC>= (aC[-1][anY-1][anX]   -  aC[0][anY-1][anX]  ))
                    &&  (mDogPC>= (aC[-1][anY-1][anX+1] -  aC[0][anY-1][anX+1]))
                    &&  (mDogPC>= (aC[-1][anY]  [anX-1] -  aC[0][anY]  [anX-1]))
                    &&  (mDogPC>= (aC[-1][anY]  [anX+1] -  aC[0][anY]  [anX+1]))
                    &&  (mDogPC>= (aC[-1][anY+1][anX-1] -  aC[0][anY+1][anX-1]))
                    &&  (mDogPC>= (aC[-1][anY+1][anX]   -  aC[0][anY+1][anX]  ))
                    &&  (mDogPC>= (aC[-1][anY+1][anX+1] -  aC[0][anY+1][anX+1]))

/*
*/
                    &&  (mDogPC>  (aC[1][anY-1][anX-1]  -  aC[2][anY-1][anX-1]))
                    &&  (mDogPC>  (aC[1][anY-1][anX]    -  aC[2][anY-1][anX]  ))
                    &&  (mDogPC>  (aC[1][anY-1][anX+1]  -  aC[2][anY-1][anX+1]))
                    &&  (mDogPC>  (aC[1][anY]  [anX-1]  -  aC[2][anY]  [anX-1]))
                    &&  (mDogPC>  (aC[1][anY]  [anX+1]  -  aC[2][anY]  [anX+1]))
                    &&  (mDogPC>  (aC[1][anY+1][anX-1]  -  aC[2][anY+1][anX-1]))
                    &&  (mDogPC>  (aC[1][anY+1][anX]    -  aC[2][anY+1][anX]  ))
                    &&  (mDogPC>  (aC[1][anY+1][anX+1]  -  aC[2][anY+1][anX+1]))

                 )
              {
                    isMax = true;
                    mResDifSift= CalculateDiff(aC,anX,anY,0);
              }
           }
           else   // mDogPC<=aLDif[-1]
           {
              if (      doMin
                    &&  (mDogPC<=aLDif[1])
                    &&  (mDogPC< aLm1[anX]-aN1m1[anX])
                    &&  (mDogPC<= aLp1[anX]-aN1p1[anX])
                    &&  (mDogPC<  aC[-1][anY][anX] -aC[0][anY][anX])
                    &&  (mDogPC<= aC[1][anY][anX] -aC[2][anY][anX])

                    &&  (mDogPC<  aLm1[anX+1]-aN1m1[anX+1])
                    &&  (mDogPC<  aLm1[anX-1]-aN1m1[anX-1])
                    &&  (mDogPC<=   aLp1[anX+1]-aN1p1[anX+1])
                    &&  (mDogPC<=   aLp1[anX-1]-aN1p1[anX-1])


                    &&  (mDogPC<   (aC[-1][anY-1][anX-1] -  aC[0][anY-1][anX-1]))
                    &&  (mDogPC<   (aC[-1][anY-1][anX]   -  aC[0][anY-1][anX]  ))
                    &&  (mDogPC<   (aC[-1][anY-1][anX+1] -  aC[0][anY-1][anX+1]))
                    &&  (mDogPC<   (aC[-1][anY]  [anX-1] -  aC[0][anY]  [anX-1]))
                    &&  (mDogPC<   (aC[-1][anY]  [anX+1] -  aC[0][anY]  [anX+1]))
                    &&  (mDogPC<   (aC[-1][anY+1][anX-1] -  aC[0][anY+1][anX-1]))
                    &&  (mDogPC<   (aC[-1][anY+1][anX]   -  aC[0][anY+1][anX]  ))
                    &&  (mDogPC<   (aC[-1][anY+1][anX+1] -  aC[0][anY+1][anX+1]))

                    &&  (mDogPC<=  (aC[1][anY-1][anX-1]  -  aC[2][anY-1][anX-1]))
                    &&  (mDogPC<=  (aC[1][anY-1][anX]    -  aC[2][anY-1][anX]  ))
                    &&  (mDogPC<=  (aC[1][anY-1][anX+1]  -  aC[2][anY-1][anX+1]))
                    &&  (mDogPC<=  (aC[1][anY]  [anX-1]  -  aC[2][anY]  [anX-1]))
                    &&  (mDogPC<=  (aC[1][anY]  [anX+1]  -  aC[2][anY]  [anX+1]))
                    &&  (mDogPC<=  (aC[1][anY+1][anX-1]  -  aC[2][anY+1][anX-1]))
                    &&  (mDogPC<=  (aC[1][anY+1][anX]    -  aC[2][anY+1][anX]  ))
                    &&  (mDogPC<=  (aC[1][anY+1][anX+1]  -  aC[2][anY+1][anX+1]))
                 )
              {
                  isMin=true;
                  mResDifSift = CalculateDiff(aC,anX,anY,0);
              }
           }
           aLDif++;
           
           if (aInteract)
           {
              mNbExtre += (isMax||isMin);
              mNbExtreOK += (isMax||isMin) && (mResDifSift==eTES_Ok);

              if (aVerifExtrema)
              {
                 bool VMin,VMax;
                 ExtramDOG(aC,Pt2di(anX,anY),VMax,VMin);

                 ELISE_ASSERT((VMin==isMin) && (VMax==isMax),"Verif Extrema : PB\n");
              }
              if (aVCD && (isMax||isMin))
              {
                 Pt2dr aPTr(anX+mTrX,anY+mTrY);
                 aVCD->SetPtsCarac
                 (
                     aPTr*mResolGlob,
                     isMax,
                     mResolGlob*mResolOctaveBase,
                     mIndexSigma,
                     mResDifSift
                 );
              }
          }
        }

   }

   if (aVerifExtrema || aInteract)
   {
      std::cout << "DOG " << mResolGlob 
             << " K-OCT " << mKInOct
             << " " << mResolOctaveBase 
             <<  " %MinMax= " << (100.0*(mNbExtre)/(mSz.x*mSz.y) ) 
             <<  " %MinMax-OK= " << (100.0*(mNbExtreOK)/(mSz.x*mSz.y) ) 
             << "\n";
   }
}

template <> INT**    cTplImInMem<U_INT1>::theMDog(0);
template <> INT**    cTplImInMem<U_INT2>::theMDog(0);
template <> INT**    cTplImInMem<INT>::theMDog(0);
template <> double** cTplImInMem<float>::theMDog(0);


InstantiateClassTplDigeo(cTplImInMem)


};



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
