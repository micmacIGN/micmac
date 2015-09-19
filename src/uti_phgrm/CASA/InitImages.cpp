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

#include "Casa.h"



void CC(
                   Video_Win *          aW,
                   Pt2di                aP0,
                   std::vector<Pt2di> & aVExpl,
                   Im2D_Bits<1> aMasq,
                   Im2D_Bits<1> aMarqTmp,
                   double aDist,
                   bool ClearTmp
             )
{
   aVExpl.clear();
   TIm2DBits<1> aTtmp(aMarqTmp);
   TIm2DBits<1> aTMasq(aMasq);
   double aD2 = ElSquare(aDist);

   aVExpl.push_back(aP0);
   aTtmp.oset(aP0,1);
   int aKCur= 0;
   while (aKCur!=int(aVExpl.size()))
   {
         Pt2di aPCur = aVExpl[aKCur];
         for (int aKV=0 ; aKV<4 ; aKV++)
         {
             Pt2di aPVois = aPCur + TAB_4_NEIGH[aKV];
             if ( 
                     (aTMasq.get(aPVois))
                  && (aTtmp.get(aPVois)==0)
                  && (square_euclid(aP0,aPVois)<aD2)
                 )
              {
                  aVExpl.push_back(aPVois);
                  aTtmp.oset(aPVois,1);
              }
         }
         aKCur++;
   }
   if (ClearTmp)
   {
      for (int aKP=0 ; aKP<int(aVExpl.size()) ; aKP++)
      {
         Pt2di aPCur = aVExpl[aKP];
         aTtmp.oset(aPCur,0);
      }
   }
}
             

cOneSurf_Casa *  cAppli_Casa::InitNuage(const cSectionLoadNuage & aSLN)
{
    cOneSurf_Casa * aRes = new cOneSurf_Casa;

    for
    (
         std::list<cNuageByImage>::const_iterator itN=aSLN.NuageByImage().begin();
         itN!=aSLN.NuageByImage().end();
         itN++
    )
    {
         AddNuage2Surf(aSLN,* itN,*aRes);
    }

    return aRes; 	
}

void cAppli_Casa::AddNuage2Surf
     (
           const cSectionLoadNuage& aSLN,
           const cNuageByImage & aNBI,
           cOneSurf_Casa& aSurf
     )
{
   std::string aNameNuage = mDC+aNBI.NameXMLNuage();
   if (  (!ELISE_fp::exist_file(aNameNuage)) && ELISE_fp::exist_file(aNBI.NameXMLNuage()))
       aNameNuage = aNBI.NameXMLNuage();



   std::string  aNameMask = "";
   if (aNBI.NameMasqSup().IsInit())
     aNameMask = mDC+aNBI.NameMasqSup().Val();

   cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromFileIm(aNameNuage,"XML_ParamNuage3DMaille",aNameMask);
   Pt2di aSz = aNuage->SzGeom();
   Im2D_Bits<1> aMasq(aSz.x,aSz.y,0);
   TIm2DBits<1> aTMasq(aMasq);

   for 
   (
      cElNuage3DMaille::tIndex2D aP= aNuage->Begin(); 
      aP != aNuage->End();
      aNuage->IncrIndex(aP)
   )
   {
         aTMasq.oset(aP,1);
   }
   ELISE_COPY(aMasq.border(1),0,aMasq.out());

   Im2D_Bits<1> aMarqTmp(aSz.x,aSz.y,0);
   TIm2DBits<1> aTtmp(aMarqTmp);
   std::vector<Pt2di> aVExpl;




    Video_Win * aW = 0;
    if ( aSLN.SzW().IsInit ()  && (aSurf.mW==0))
    {
        Pt2di aSzW= aSLN.SzW().Val();
        double aRatio = ElMin(aSzW.x/double(aSz.x),aSzW.y/double(aSz.y));
        aSzW = round_ni(Pt2dr(aSz)*aRatio);
        aW = Video_Win::PtrWStd(aSzW,1);
        aW= aW->PtrChc(Pt2dr(0,0),Pt2dr(aRatio,aRatio));
        aSurf.mW = aW;
    }
    aW =  aSurf.mW ;
    if (aSurf.mW) 
       ELISE_COPY(aMasq.all_pts(),aMasq.in(),aSurf.mW->odisc());

    std::list<Pt2di>  aLGerm;
    

 // {
 // Calcule tous les germe de composante connexe dans aLGerm
 // aTmp sert de marqueur, a la fin il estr remis a zero
 // le parcourt est limite a DistSep
    Pt2di aP0;
    for (aP0.x=0 ; aP0.x<aSz.x; aP0.x++)
    {
        for (aP0.y=0 ; aP0.y<aSz.y; aP0.y++)
        {
            if (aTMasq.get(aP0) && (!aTtmp.get(aP0)))
            {
               // if (aW) aW->draw_circle_abs(aP0,1,aW->pdisc()(P8COL::green)); 
               aLGerm.push_back(aP0);
               CC(aW,aP0,aVExpl,aMasq,aMarqTmp,aSLN.DistSep().Val(),false);
            }
        }
    }
    ELISE_COPY(aMarqTmp.all_pts(),0,aMarqTmp.out());
  //}



    // Parcourt 2 fois les germes, reexplore les meme composantes connexes que + haut
    //   - la premiere fois calcul une image Cpt permettant de savoir pour chaque point
    //   da combien de zone il appartient
    //   - la deuxieme fois  accumule chaque point dans le facetton
    std::vector<cFaceton> & aVF = aSurf.mVF;

    Im2D_U_INT1 aImCpt(aSz.x,aSz.y,0);
    TIm2D<U_INT1,INT> aTCpt(aImCpt);
    for (int aTime=0 ; aTime<2 ; aTime++)
    {
        
        for 
        (
            std::list<Pt2di>::const_iterator iTp=aLGerm.begin();
            iTp!=aLGerm.end();
            iTp++
        )
        {
             cAccumFaceton anAcu;
             CC(aW,*iTp,aVExpl,aMasq,aMarqTmp,aSLN.DistZone().Val(),true);
             
             for (int aKP=0 ; aKP<int(aVExpl.size()) ; aKP++)
             {
                Pt2di aP = aVExpl[aKP];
                if (aTime==0)
                {
                    aTCpt.oset(aP,ElMin(255,1+aTCpt.get(aP)));
                }
                else
                {
                     anAcu.Add
                     (
                         Pt2dr(aP),
                         aNuage->PtOfIndex(aP),
                         1.0/double(aTCpt.get(aP))
                     );
                     // if (aW) aW->draw_circle_abs(aP,1,aW->pdisc()(P8COL::red));
                }
             }
             if (aTime!=0)
             {
                cFaceton aFct = anAcu.CompileF(*aNuage);
                if (aFct.Ok())
                {
                    aVF.push_back(aFct);
                    if (aW)
                    {
                        Pt2dr aP0 =aFct.Index();
                        Pt3dr aN = aFct.Normale();
                        if (aN.z<0) aN = aN*-1;
                        Pt2dr aDir(aN.x,aN.y);
                        aW->draw_circle_abs(aP0,1,aW->pdisc()(P8COL::green)); 
                        aW->draw_seg(aP0,aP0+aDir*30,aW->pdisc()(P8COL::red)); 
                    }
                }
                // getchar();
             }
        }
      std::cout << "For " << aNBI.NameXMLNuage() << " Iter " << aTime << " Loaded \n";
    }
    if (aW) 
      getchar();
    delete aNuage;
}

void cAppli_Casa::EstimSurf
     (
         cOneSurf_Casa & aSurf,
         const cSectionEstimSurf & aSES
     )
{

    Video_Win * aW = aSurf.mW;
    std::vector<cFaceton> & aVF = aSurf.mVF;

    int aKMoy = cFaceton::GetIndMoyen(aVF);
    aSurf.mFMoy = &aVF[aKMoy];
    
    if (aW)
    {
       aW->draw_circle_abs(aVF[aKMoy].Index(),6,aW->pdisc()(P8COL::red)); 
    }

    switch (aSES.TypeSurf())
    {
        case eTSA_CylindreRevolution :
            EstimeCylindreRevolution(aSurf,aSES);
        break;
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
Footer-MicMac-eLiSe-25/06/2007*/
