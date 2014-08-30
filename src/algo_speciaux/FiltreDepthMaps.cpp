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
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/


#include "StdAfx.h"
#include "hassan/reechantillonnage.h"


/***********************************************************************************/
/*                                                                                 */
/*                      FiltreDetecRegulProf                                       */
/*                                                                                 */
/***********************************************************************************/

class cCC_NbMaxIter : public  cCC_NoActionOnNewPt
{
   public  :
       cCC_NbMaxIter(int aNbMax) :
          mNbIter (0),
          mNbMaxIter (aNbMax)
       {
       }


       void OnNewStep() { mNbIter++;}
       void  OnNewPt(const Pt2di & aP) 
       {
           mVPts.push_back(aP);
       }
       bool  StopCondStep() {return mNbIter>=mNbMaxIter;}

    
       std::vector<Pt2di> mVPts;
       int                mNbIter;
       int                mNbMaxIter;
};


class cCCMaxAndBox : public  cCC_NbMaxIter
{
   public  :
       cCCMaxAndBox(int aNbMax,const Box2di & aBox) :
           cCC_NbMaxIter(aNbMax),
           mBox(aBox)
       {
       }
       bool ValidePt(const Pt2di & aP){return mBox.inside(aP);}

       Box2di mBox;
};


template <class tNum,class tNBase>  Im2D_Bits<1> TplFiltreDetecRegulProf
                                        (
                                             TIm2D<tNum,tNBase> aTProf, 
                                             TIm2DBits<1>  aTMasq,
                                             const cParamFiltreDetecRegulProf & aParam
                                        )
{
    FiltrageCardCC(true,aTMasq,1,0, aParam.NbCCInit().Val());

    Pt2di aSz = aTProf.sz();
    Im2D_Bits<1> aIMasq = aTMasq._the_im;

    Im2D_Bits<1> aMasqTmp = ImMarqueurCC(aSz);
    TIm2DBits<1> aTMasqTmp(aMasqTmp);
    bool V4= aParam.V4().Val();

    Im2D_REAL4 aImDif(aSz.x,aSz.y);
    TIm2D<REAL4,REAL8> aTDif(aImDif);

    ELISE_COPY(aIMasq.border(1),0,aIMasq.out());

    Pt2di aP;
    int aSzCC = aParam.SzCC().Val();
    double aPondZ = aParam.PondZ().Val();
    double aPente = aParam.Pente().Val();
    for (aP.x =0 ; aP.x < aSz.x ; aP.x++)
    {
        for (aP.y =0 ; aP.y < aSz.y ; aP.y++)
        {
             if (aTMasq.get(aP))
             {
                 cCC_NbMaxIter aCCParam(aSzCC);
                 OneZC(aP,V4,aTMasqTmp,1,0,aTMasq,1,aCCParam);
                 tNBase aZ0 =  aTProf.get(aP);
                 double aSomP = 0;
                 int    aNbP  = 0;
                 for (int aKP=0 ; aKP<int(aCCParam.mVPts.size()) ; aKP++)
                 {
                     const Pt2di & aQ = aCCParam.mVPts[aKP];
                     aTMasqTmp.oset(aQ,1);
                     if (aKP>0)
                     {
                         double aDist = euclid(aP,aQ);
                         double aDZ = ElAbs(aZ0-aTProf.get(aQ));
                         double aAttZ = aPondZ + aPente * aDist;
                         double aPds  = 1 / (1 + ElSquare(aDZ/aAttZ));
                         aNbP++;
                         aSomP += aPds;
                     }
                 }
                 aNbP = ElMax(aNbP,aSzCC*(1+aSzCC));
                 aTDif.oset(aP,aSomP/aNbP);
             }
        }
    }
    if (aParam.NameTest().IsInit())
    {
       Tiff_Im::Create8BFromFonc(aParam.NameTest().Val(),aSz,aImDif.in()*255);
    }

    Im2D_Bits<1> aIResult(aSz.x,aSz.y);
    ELISE_COPY(aIResult.all_pts(),(aImDif.in()> aParam.SeuilReg().Val()) && (aIMasq.in()) , aIResult.out());
    return aIResult;
    
}

Im2D_Bits<1>  FiltreDetecRegulProf(Im2D_REAL4 aImProf,Im2D_Bits<1> aIMasq,const cParamFiltreDetecRegulProf & aParam)
{
   return TplFiltreDetecRegulProf(TIm2D<REAL4,REAL8>(aImProf),TIm2DBits<1>(aIMasq),aParam);
}


void TestFiltreRegul()
{
   Pt2di aP0(2000,500);
   Pt2di aSz(500,500);

   Video_Win * aW = 0;


   Tiff_Im aFileProf ("/home/marc/TMP/EPI/EXO1-Fontaine/MTD-Image-CIMG_2489.JPG/Fusion_NuageImProf_LeChantier_Etape_1.tif");
   Tiff_Im aFileMasq ("/home/marc/TMP/EPI/EXO1-Fontaine/MTD-Image-CIMG_2489.JPG/Fusion_NuageImProf_LeChantier_Etape_1_Masq.tif");

   Im2D_REAL4    aImProf(aSz.x,aSz.y);
   Im2D_Bits<1>  aMasq(aSz.x,aSz.y);


   ELISE_COPY(aImProf.all_pts(),trans(aFileProf.in(0),aP0),aImProf.out());
   ELISE_COPY(aMasq.all_pts(),trans(aFileMasq.in(0),aP0),aMasq.out());

   if (aW)
   {
       ELISE_COPY(aImProf.all_pts(),aImProf.in()*5,aW->ocirc());
       ELISE_COPY(select(aMasq.all_pts(),!aMasq.in()),P8COL::black,aW->odisc());
   }

   cParamFiltreDetecRegulProf aParam;
   //TplFiltreDetecRegulProf(TIm2D<REAL4,REAL8>(aImProf),TIm2DBits<1>(aMasq),aParam);
std::cout << "AAAaaaA\n";
   FiltreDetecRegulProf(aImProf,aMasq,aParam);
std::cout << "BBBbBb\n";
getchar();
}

/***********************************************************************************/
/*                                                                                 */
/*                      ReduceImageProf                                            */
/*                                                                                 */
/***********************************************************************************/

template <class tNum,class tNBase>  Im2D_REAL4   TplFReduceImageProf
                                        (
                                             Im2D_REAL4    aImPds,
                                             TIm2DBits<1>  aTMasq,
                                             TIm2D<tNum,tNBase> aTProf, 
                                             double aScale

                                        )
{
   double aSeuilDif = 0.1;
std::cout << "IInnnnnnnnnnnnnnnnnnnnn\n";
    TIm2D<REAL4,REAL8> aTPds(aImPds);
    Pt2di aSzOut = aImPds.sz();
    Im2D<tNum,tNBase> aIProf = aTProf._the_im;
    Pt2di aSzIn = aTProf.sz();
    Im2D_REAL4 aRes(aSzOut.x,aSzOut.y,0.0);
    TIm2D<REAL4,REAL8> aTRes(aRes);

    Im2D_Bits<1> aMasqTmpCC = ImMarqueurCC(aSzIn);
    TIm2DBits<1> aTMasqTmpCC(aMasqTmpCC);


    Im2D_Bits<1> aMasqTmpIn(aSzIn.x,aSzIn.y,0);
    TIm2DBits<1> aTMIn(aMasqTmpIn);

    if (aScale==1)
    {
        ELISE_COPY(aRes.all_pts(),aIProf.in(),aRes.out());

        return aRes;
    }
    int aSzCC = ElMax(3,round_up(aScale*2+1));

    Pt2di aPOut;
    for (aPOut.x = 0 ; aPOut.x<aSzOut.x ; aPOut.x++)
    {
        for (aPOut.y = 0 ; aPOut.y<aSzOut.y ; aPOut.y++)
        {
              aTPds.oset(aPOut,0.0);
              Pt2dr aPRIn (aPOut.x*aScale,aPOut.y*aScale);
              int aXInCentreI = ElMax(1,ElMin(aSzIn.x-2,round_ni(aPRIn.x)));
              int aYInCentreI = ElMax(1,ElMin(aSzIn.y-2,round_ni(aPRIn.y)));

              int aXI0 = ElMax(0,aXInCentreI-aSzCC);
              int aYI0 = ElMax(0,aYInCentreI-aSzCC);
              int aXI1 = ElMin(aSzIn.x-2,aXInCentreI+aSzCC);
              int aYI1 = ElMin(aSzIn.y-2,aYInCentreI+aSzCC);

              // 1 calcul du barrycentre
              Pt2dr aBar(0,0);
              double aSomP=0;
              Pt2di aPIn;
              for (aPIn.x=aXI0 ; aPIn.x<=aXI1 ; aPIn.x++)
              {
                  for (aPIn.y=aYI0 ; aPIn.y<=aYI1 ; aPIn.y++)
                  {
                       if (aTMasq.get(aPIn))
                       {
                            aSomP++;
                            aBar = aBar + Pt2dr(aPIn);
                       }
                  }
              }
              if (aSomP>0)
              {
                  // 1 calcul du pt le plus pres du barrycentre
                  aBar = aBar / double(aSomP);
                  double aDistMin = 1e9;
                  Pt2di aNearest(-1,-1);
                  for (aPIn.x=aXI0 ; aPIn.x<=aXI1 ; aPIn.x++)
                  {
                      for (aPIn.y=aYI0 ; aPIn.y<=aYI1 ; aPIn.y++)
                      {
                           if (aTMasq.get(aPIn))
                           {
                                double aDist = square_euclid(Pt2dr(aPIn)-aPRIn);
                                if (aDist<aDistMin)
                                {
                                    aDist = aDistMin;
                                    aNearest = aPIn;
                                }
                           }
                      }
                  }
                  cCCMaxAndBox  aCCParam(aSzCC,Box2di(Pt2di(aXI0,aYI0),Pt2di(aXI1,aYI1)));
                  OneZC(aNearest,true,aTMasqTmpCC,1,0,aTMasq,1,aCCParam);

                  const std::vector<Pt2di> aVP = aCCParam.mVPts;
                  int aNbP = aVP.size();

                  for (int aKP=0 ; aKP<aNbP ; aKP++)
                  {
                     aTMIn.oset(aVP[aKP],1);
                     aTMasqTmpCC.oset(aVP[aKP],1);
                  }

                  Pt2di aPIn;
                  std::vector<double> aVProf;
                  std::vector<double> aVDif;
                  for (aPIn.x=aXI0 ; aPIn.x<=aXI1 ; aPIn.x++)
                  {
                      for (aPIn.y=aYI0 ; aPIn.y<=aYI1 ; aPIn.y++)
                      {
                           if (aTMIn.get(aPIn))
                           {
                               double aProf = aTProf.get(aPIn);
                               aVProf.push_back(aProf);

                               Pt2di aPx = aPIn + Pt2di(1,0);
                               if (aTMIn.get(aPx))
                               {
                                   aVDif.push_back(ElAbs(aProf-aTProf.get(aPx)));
                               }
                          
                               Pt2di aPy = aPIn + Pt2di(0,1);
                               if (aTMIn.get(aPy))
                               {
                                   aVDif.push_back(ElAbs(aProf-aTProf.get(aPy)));
                               }
                           }
                      }
                  }
                  double aProfRef = aTProf.get(aNearest);
                  if (aVDif.size()>3)
                  {
                     double aDifStd = ElMax(MedianeSup(aVDif),aSeuilDif);
                     double aSomPds = 0;
                     double aSomProf = 0;
                     for (int aKP=0 ; aKP< aNbP ; aKP++)
                     {
                         const Pt2di & aP = aVP[aKP];
                         double aDist= euclid(aP-aNearest);
                         double aProf =  aTProf.get(aP);
                         double aDifNorm = ElAbs(aProf-aProfRef) /(aDifStd * (1+aDist/2.0));
                         double aPdsProf = 1.0;
                         if (aDifNorm<1)
                         {
                         }
                         else if (aDifNorm<3)
                         {
                             aPdsProf = (3-aDifNorm) /2.0;
                         }
                         else
                         {
                             aPdsProf = 0;
                         }
                         double aDistNorm= ElMin(aDist/aScale,2.0);
                         double aPdsDist = (1+cos(aDistNorm * (PI/2.0)));

                         double aPds = aPdsDist * aPdsProf;
                         aSomPds += aPds;
                         aSomProf += aPds*aProf;
                     }
                     aTPds.oset(aPOut,1.0);
                     aTRes.oset(aPOut,aSomProf/aSomPds);
                  }
                  else
                  {
                     aTPds.oset(aPOut,1.0);
                     aTRes.oset(aPOut,aProfRef);
                     // getchar();
                  }

                  for (int aKP=0 ; aKP<aNbP ; aKP++)
                     aTMIn.oset(aVP[aKP],0);
              }
/*
*/
        }
    }

    std::cout << "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqq " << aScale << aSzOut << aSzIn << "\n";
   
    return aRes;
}

Im2D_REAL4 ReduceImageProf(Im2D_REAL4 aImPds,Im2D_Bits<1> aIMasq,Im2D_REAL4 aImProf,double aScale)
{
   return TplFReduceImageProf(aImPds,TIm2DBits<1>(aIMasq),TIm2D<REAL4,REAL8>(aImProf),aScale);
}
Im2D_REAL4 ReduceImageProf(Im2D_REAL4 aImPds,Im2D_Bits<1> aIMasq,Im2D_INT2 aImProf,double aScale)
{
   return TplFReduceImageProf(aImPds,TIm2DBits<1>(aIMasq),TIm2D<INT2,INT>(aImProf),aScale);
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
