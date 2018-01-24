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

cSetPCarac * LoadStdSetCarac(const std::string & aNameIm,const std::string & aExt)
{
   return new cSetPCarac
               (
                   StdGetObjFromFile<cSetPCarac>
                   (
                        NameFileNewPCarac(aNameIm,true, aExt),
                        MMDir() + "src/uti_image/NewRechPH/ParamNewRechPH.xml",
                        "SetPCarac",
                        "SetPCarac"
                    )
               );
}


// Visualise une conversion de flux en vecteur de point
void  TestFlux2StdCont()
{
    std::vector<Pt2di> aVp;
    Flux2StdCont(aVp,circle(Pt2dr(60,60),50));
    Im2D_U_INT1 aIm(120,120,0);
    for (int aK=0; aK<int(aVp.size()) ; aK++)
    {
        Pt2di aP = aVp[aK];
        aIm.data()[aP.y][aP.x] = 1;
    }
    Video_Win  aW =  Video_Win::WStd(Pt2di(300,200),3);
    ELISE_COPY(aIm.all_pts(),aIm.in(),aW.odisc());
    getchar();

}

/*
class cSurfQuadr
{
    public :
    private :
               
};
*/


/*****************************************************/
/*                                                   */
/*                 ::                                */
/*                                                   */
/*****************************************************/

class cCmpPt2diOnEuclid
{
   public : 
       bool operator () (const Pt2di & aP1, const Pt2di & aP2)
       {
                   return square_euclid(aP1) < square_euclid(aP2) ;
       }
};

std::vector<Pt2di> SortedVoisinDisk(double aDistMin,double aDistMax,bool Sort)
{
   std::vector<Pt2di> aResult;
   int aDE = round_up(aDistMax);
   Pt2di aP;
   for (aP.x=-aDE ; aP.x <= aDE ; aP.x++)
   {
       for (aP.y=-aDE ; aP.y <= aDE ; aP.y++)
       {
            double aD = euclid(aP);
            if ((aD <= aDistMax) && (aD>aDistMin))
               aResult.push_back(aP);
       }
   }
   if (Sort)
   {
      cCmpPt2diOnEuclid aCmp;
      std::sort(aResult.begin(),aResult.end(),aCmp);
   }

   return aResult;
}

Pt3di Ply_CoulOfType(eTypePtRemark aType,int aL0,int aLong)
{
    if (aLong==0) 
       return Pt3di(255,255,255);


    double aSeuil = 5.0;
    if (aLong < 5) 
    {
       int aG = 255 * ( aSeuil - aLong) / aSeuil;
       return Pt3di(aG,aG,aG);
    }



    switch(aType)
    {
         case eTPR_LaplMax  : return Pt3di(255,128,128);
         case eTPR_LaplMin  : return Pt3di(128,128,255);

         case eTPR_GrayMax  : return Pt3di(255,  0,  0);
         case eTPR_GrayMin  : return Pt3di(  0,  0,255);
         case eTPR_GraySadl : return Pt3di(  0,255,  0);

         default :;
    }

    return  Pt3di(128,128,128);
}

Pt3dr X11_CoulOfType(eTypePtRemark aType)
{
   Pt3di aCI = Ply_CoulOfType(aType,0,1000);
   return Pt3dr(aCI) /255.0;
}

void ShowPt(const cOnePCarac & aPC,const ElSimilitude & aSim,Video_Win * aW,bool HighLight)
{
    if (! aW) return;

    Pt3dr aC = X11_CoulOfType(aPC.Kind());
    Col_Pal aCPal = aW->prgb()(aC.x*255,aC.y*255,aC.z*255);

    aW->draw_circle_abs(aSim(aPC.Pt()),3.0,aCPal);
    if (HighLight)
    {
       aW->draw_circle_abs(aSim(aPC.Pt()),5.0,aCPal);
       aW->draw_circle_abs(aSim(aPC.Pt()),7.0,aCPal);
    }
}


/*****************************************************/
/*                                                   */
/*                  cPtRemark                        */
/*                                                   */
/*****************************************************/

cPtRemark::cPtRemark(const Pt2dr & aPt,eTypePtRemark aType,int aNiv) :
     mPtR   (aPt),
     mType  (aType),
     mHRs   (),
     mLR    (0),
     mNiv   (aNiv)
{
}


void cPtRemark::MakeLink(cPtRemark * aHR)
{
   mHRs.push_back(aHR);
   aHR->mLR = this;
}

void  cPtRemark::RecGetAllPt(std::vector<cPtRemark *> & aRes)
{
     aRes.push_back(this);
     for (auto & aPt:  mHRs)
         aPt->RecGetAllPt(aRes);
}

/*****************************************************/
/*                                                   */
/*                  cBrinPtRemark                    */
/*                                                   */
/*****************************************************/

int SignOfType(eTypePtRemark aKind)
{
   switch(aKind)
   {
       case eTPR_LaplMax :
       case eTPR_GrayMax :
            return 1;

       case eTPR_LaplMin :
       case eTPR_GrayMin :
            return -1;
       default :
            ELISE_ASSERT(false,"cAppli_NewRechPH::PtOfBuf");
   }
   return 0;
}

cBrinPtRemark::cBrinPtRemark(cPtRemark * aLR,cAppli_NewRechPH & anAppli) :
    mLR        (aLR),
    mScaleStab (-1)
{
    std::vector<cPtRemark *> aVPt;
    mLR->RecGetAllPt(aVPt);
    
    int aNbMult=0;
    int aNivMin = aLR->Niv();

    for (auto & aPt:  aVPt)
    {
        aNivMin = ElMin(aNivMin,aPt->Niv());
        if (aPt->HRs().size()>=2)
           aNbMult++;
    }

    mOk = (aNbMult==0) && anAppli.OkNivStab(aLR->Niv()) && (aNivMin==0);
    if (!mOk) return;

    mScaleStab = anAppli.ScaleOfNiv(aLR->Niv());

    int aSign = SignOfType(mLR->Type());
    std::vector<double> aVLapl;
    mLaplMax = -1;

    for (auto & aPt:  aVPt)
    {
        int aNiv = aPt->Niv();
        if (anAppli.OkNivLapl(aNiv))
        {
           double aLapl =  0;

          if (anAppli.ScaleCorr())
          {
              aLapl = anAppli.GetImOfNiv(aNiv)->QualityScaleCorrel(round_ni(aPt->Pt()),aSign,true)  ;
          }
          else
          {
              aLapl =  anAppli.GetLapl(aNiv,round_ni(aPt->Pt()),mOk) * aSign;
          }
 
           if (!mOk)
           {
               return;
           }

           if (aLapl> mLaplMax)
           {
               mNivScal = aNiv;
               mScale   =  anAppli.ScaleOfNiv(aNiv);
               mLaplMax = aLapl;
           }
        }
      //   std::cout << "CORRELL " << anAppli.GetImOfNiv(aNiv)->QualityScaleCorrel(round_ni(aPt->Pt()),aSign,true)  << " \n";
    }
    
    // getchar();
    // std::cout << "SSsSSsS= " <<  aLaplMax << "\n";

/*
    static int aNbBr= 0;
    static int aNbOk=0;
    aNbBr++;
    aNbOk += aNbBr;
*/
}

std::vector<cPtRemark *> cBrinPtRemark::GetAllPt()
{
    std::vector<cPtRemark *> aRes;
    mLR->RecGetAllPt(aRes);
    return aRes;
}



/*
cPtRemark *  cBrinPtRemark::Nearest(int & aNivMin,double aTargetNiv)
{
    cPtRemark * aRes = mP0;
    cPtRemark * aPtr = aRes;
    int aCurNiv = mNiv0;
    aNivMin = aCurNiv;
    double aScoreMin = 1e10;
    while (aPtr)
    {
       double aScore = ElAbs(aTargetNiv-aCurNiv);
       if (aScore < aScoreMin)
       {
           aScore = aScoreMin;
           aRes = aPtr;
           aNivMin = aCurNiv;
       }
       aPtr = aPtr->LR();
       aCurNiv ++;
    }
    return aRes;
}
*/




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
