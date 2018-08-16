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

std::string NameFileNewPCarac(eTypePtRemark aLab,const std::string & aNameGlob,bool Bin,const std::string & anExt)
{
    std::string aDirGlob = DirOfFile(aNameGlob);
    std::string aDirLoc= "NewPH" + anExt + "/";
    ELISE_fp::MkDirSvp(aDirGlob+aDirLoc);

    return aDirGlob+aDirLoc + NameWithoutDir(aNameGlob) +  "_" + eToString(aLab) + (Bin ? ".dmp" : ".xml");
}


cSetPCarac * LoadStdSetCarac(eTypePtRemark aLab,const std::string & aNameIm,const std::string & aExt)
{
   if (aLab != eTPR_NoLabel)
   {
      return new cSetPCarac(StdGetFromNRPH(NameFileNewPCarac(aLab,aNameIm,true, aExt),SetPCarac));
   }

   cSetPCarac * aRes = new cSetPCarac;
   for (int aKLab=0 ; aKLab<int(eTPR_NoLabel) ; aKLab++)
   {
       cSetPCarac * aSetLab =  new cSetPCarac(StdGetFromNRPH(NameFileNewPCarac(eTypePtRemark(aKLab),aNameIm,true, aExt),SetPCarac));
       for (auto & aPC : aSetLab->OnePCarac())
       {
           aRes->OnePCarac().push_back(aPC);
       }
       delete aSetLab;
   }
     
   return aRes;
}

void  SaveStdSetCaracMultiLab(const cSetPCarac aSetGlob,const std::string & aNameIm,const std::string & aExt,int aSeuilHS)
{
   for (int aKLab=0 ; aKLab<int(eTPR_NoLabel) ; aKLab++)
   {
      cSetPCarac  aSetLab ;
      eTypePtRemark aLab = (eTypePtRemark) aKLab;
      for (auto & aPC : aSetGlob.OnePCarac())
      {
         if (aPC.Kind() == aLab)
         {
            aSetLab.OnePCarac().push_back(aPC);
         }
      }
      if (aSeuilHS<0)
      {
         MakeFileXML(aSetLab,NameFileNewPCarac(aLab,aNameIm,true,aExt));
      }
      else
      {
          std::vector<double> aVScale;
          for (auto & aPC : aSetLab.OnePCarac())
              aVScale.push_back(ScaleGen(aPC));
          int aKSeuil = ElMax(0, int(aVScale.size()-1)-aSeuilHS);
// std::cout << "aScaleLimaScaleLim " << aVScale.size() << " " << aKSeuil << "\n";
          double aScaleLim = (aVScale.empty()) ? 0.0 : KthVal(aVScale,aKSeuil);
// std::cout << "********** aScaleLimaScaleLim \n";
          cSetPCarac aSetHighS;
          cSetPCarac aSetLowS;
          for (auto & aPC : aSetLab.OnePCarac())
          {
              if (ScaleGen(aPC) >= aScaleLim)
              {
                 aSetHighS.OnePCarac().push_back(aPC);
              }
              else
              {
                 aSetLowS.OnePCarac().push_back(aPC);
              }
          }
/*
std::cout << "HHHhhhhhh " << aSetLab.OnePCarac().size() << " " 
                          <<  aSetHighS.OnePCarac().size() << " " 
                          << aSetLowS.OnePCarac().size() << "\n" ;
*/
          MakeFileXML(aSetLowS,NameFileNewPCarac(aLab,aNameIm,true,aExt));
          MakeFileXML(aSetHighS,NameFileNewPCarac(aLab,aNameIm,true,"_HighS"+aExt));
      }
   }
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
     mRPt   (aPt),
     mType  (aType),
     mHighRs (),
     mLowR  (0),
     mNiv   (aNiv)
{
}


void cPtRemark::MakeLink(cPtRemark * aHighRes)
{
   mHighRs.push_back(aHighRes);
   aHighRes->mLowR = this;
}

void  cPtRemark::RecGetAllPt(std::vector<cPtRemark *> & aRes)
{
     aRes.push_back(this);
     for (auto & aPt:  mHighRs)
         aPt->RecGetAllPt(aRes);
}


Pt2dr cPtRemark::RPtAbs(cAppli_NewRechPH & anAppli) const
{
    cOneScaleImRechPH *  anIm = anAppli.GetImOfNiv(mNiv,true);
    return mRPt * (double) anIm->PowDecim();
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
       case eTPR_BifurqMax :
            return 1;

       case eTPR_LaplMin :
       case eTPR_GrayMin :
       case eTPR_BifurqMin :
            return -1;
       case eTPR_GraySadl :
            return 0;
       default :
            ELISE_ASSERT(false,"cAppli_NewRechPH::PtOfBuf");
   }
   return 0;
}

cBrinPtRemark::cBrinPtRemark(cPtRemark * aLR,cAppli_NewRechPH & anAppli) :
    mLR          (aLR),
    mBrScaleStab (-1),
    mBifurk      (nullptr)
{
    std::vector<cPtRemark *> aVPt;
    mLR->RecGetAllPt(aVPt);
    
    int aNbMult=0;
    int aNivMin = aLR->Niv();

    for (auto & aPt:  aVPt)
    {
        aNivMin = ElMin(aNivMin,aPt->Niv());
        if (aPt->HighRs().size()>=2)
        {
           aNbMult++;
           mBifurk = aPt;
        }
    }

    if (aNbMult==1) 
    {
       mOk = true;
       mNivScal =  mBifurk->Niv();
       mScale =  anAppli.ScaleAbsOfNiv(mNivScal);
       mBrScaleStab =  mScale;
       mScaleNature =  mScale;
       return;
       // static int aCpt=0; aCpt++;
       // std::cout << "PMUuUL ====== " << aCpt << "=======================================================================\n";
    }
    mBifurk = nullptr;

    mOk = (aNbMult==0) && anAppli.OkNivStab(aLR->Niv()) && (aNivMin==0);
    if (!mOk) return;

    mBrScaleStab = anAppli.ScaleAbsOfNiv(aLR->Niv());


/*
    bool  Debug = (mBrScaleStab >=15.0);
    if (Debug)
    {
        Pt2dr aPt =  aLR->RPtAbs(anAppli);
        std::cout << "SSSS mBrScaleStab " << mBrScaleStab << " " << aPt <<  " " << (aPt/512.0) << "\n";
    }
*/

// std::cout << "aLR-aLR-aLR-aLR- NIIIV " << aLR->Niv() << "\n";

    int aSign = SignOfType(mLR->Type());

    if (aSign)
    {
        std::vector<double> aVLapl;
        mLaplMax = -1;
        mLaplMaxNature = -1;

        for (auto & aPt:  aVPt)
        {
            int aNiv = aPt->Niv();
            if (anAppli.OkNivLapl(aNiv))
            {
               double aLapl =  0;

              if (anAppli.ScaleCorr())
              {
                  aLapl = anAppli.GetImOfNiv(aNiv,true)->QualityScaleCorrel(round_ni(aPt->RPt()),aSign,true)  ;
              }
              else
              {
                  aLapl =  anAppli.GetLapl(aNiv,round_ni(aPt->RPt()),mOk) * aSign;
              }
 
               if (!mOk)
               {
                   return;
               }

               double aScale = anAppli.ScaleAbsOfNiv(aNiv);
               if ((aLapl>mLaplMax)  && anAppli.ScaleIsValid(aScale))
               {
                   mNivScal = aNiv;
                   mScale   =  aScale;
                   mLaplMax = aLapl;
               }
               if (aLapl>mLaplMaxNature) 
               {
                    mScaleNature = mScale;
                    mLaplMaxNature = aLapl;
               }
            }
      //   std::cout << "CORRELL " << anAppli.GetImOfNiv(aNiv)->QualityScaleCorrel(round_ni(aPt->Pt()),aSign,true)  << " \n";
        }

        if (mLaplMax==-1)
        {
           mOk = false;
           return ;
        }
   }
   else
   {
         mNivScal =  mLR->Niv();
         mScale =  mBrScaleStab;
         mScaleNature =  mBrScaleStab;
   }
    
/*
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

/*****************************************************/
/*                                                   */
/*                  cFHistoInt                       */
/*                                                   */
/*****************************************************/

cFHistoInt::cFHistoInt() :
   mSom (0)
{
}

int cFHistoInt::at(int aK)
{
   return ((aK>=0) && (aK<int(mHist.size()))) ? mHist.at(aK) : 0;
}

void cFHistoInt::Add(int aK,double aPds,int aLine)
{
   if (aK<0)
   {
       std::cout << "KKKKK = " << aK << " LINE=" << aLine<< "\n";
       ELISE_ASSERT(aK>=0,"cFHistoInt::Add");
   }
   mSom++;
   while(int(mHist.size())<= aK)
     mHist.push_back(0);
   mHist.at(aK) += aPds;
}

double cFHistoInt::Perc(int aK)
{
   return mHist.at(aK) * (100.0/mSom);
}


void cFHistoInt::Show()
{
   double aSomPond = 0.0;
   for (int aK=0 ; aK<int(mHist.size()) ; aK++)
   {
       if (at(aK))
       {
          aSomPond += aK * at(aK);
          std::cout << " Hist " << aK << " %=" << Perc(aK)  << " Nb=" << at(aK) << "\n";
       }
   }
   std::cout << " HistMoy= " << aSomPond / mSom << "\n";
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
